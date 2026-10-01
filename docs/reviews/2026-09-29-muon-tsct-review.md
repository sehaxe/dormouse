# muon-tsct — two independent reviews, one lane

Merged 2026-10-01 from the `muon-tsct-review-a.md` and `muon-tsct-review-b.md` halves, both at
`5cfdbda`, which is where a reader finds each one whole. The halves were written independently
and neither read the other (A: Independent review of `docs/papers/muon-plus.md` and `docs/papers/tsct.md`; B: Review B — muon-plus.md / tsct.md, engineering-consequence pass),
which is why both verdicts are kept: where they agree the finding is settled,
where they disagree the disagreement is the finding. Nothing was reworded.

## Reviewer A — Independent review of `docs/papers/muon-plus.md` and `docs/papers/tsct.md`

**Reviewer pass, 2026-09-29.** Read-only. No GPU, no build, no test. Primary sources
fetched directly (arXiv HTML for `2602.21545v1/v3` and `2604.00733`, GitHub API for
`K1seki221/MuonPlus`); code read in-tree. One file written; nothing else touched.

**Headline:** claim 3 is **confirmed and proved** (the branch is unsatisfiable, no
counterexample exists). Claim 5's **verdict survives but its evidence is factually
wrong** — the paper *does* print the coefficients, in v3, and one of them is the
exact triple `dormouse-spectral` uses. Claim 7's dependency half is **refuted**. The
report's biggest miss is a live, unguarded, 112-host-sync-per-step path it found but
did not count.

---

### 0. Verdict table

| # | claim | verdict |
|---|---|---|
| 1 | every arXiv id cited resolves | **TRUE** — 10/10 HTTP 200, titles match |
| 2 | `github.com/K1seki221/MuonPlus` is the paper's code and exists | **TRUE** |
| 3 | `dormouse-muon-plus/src/lib.rs:146` `if nc*4 < nr` is unsatisfiable; the factored branch and `ns_combine_cuda` are dead | **TRUE — proved below** |
| 4 | the `3.6x on [8192,512]` comment is backwards on FLOPs | **TRUE in direction, report's own number is wrong** (1.4545x, not 1.48x) |
| 5 | `dormouse-spectral:164-166` mis-cites Muon+ §1; the paper "prints no NS coefficients" | **VERDICT TRUE, EVIDENCE FALSE** — v3 App. D.1 and D.3 print both coefficient families, D.3 terminating at exactly `(1.875, −1.25, 0.375)` |
| 6 | TSCT has never been shown to help | **TRUE** |
| 7 | `dormouse-sct` is a declared dependency with zero call sites; `param.rs:1` stale | **SPLIT** — `param.rs:1` stale: TRUE. Dead dependency: **FALSE** (it is a `[dev-dependencies]` entry and `examples/tsct_diag.rs` uses it) |
| 8 | `retract_batched` is called only from tests | **TRUE** — and the report understates the consequence by ~2 orders of magnitude |
| 9 | the `37%` is v3-only and is time-to-target-loss | **TRUE** — and the string is in the *crate* README, not the project README |

---

### 1. Claim 3 — the proof

**CONFIRMED. The condition is unsatisfiable. No counterexample exists, and none
can exist.**

`vendor/dormouse-fused/crates/dormouse-muon-plus/src/lib.rs:127-146`:

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

### 2. Claim 4 — the FLOP arithmetic, redone

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

### 3. Claim 5 — I read the paper. The report's evidence is wrong.

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
`dormouse-spectral/src/lib.rs:164-166` is real — a mis-citation is a mis-citation under
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

### 4. Claims 1, 2, 6, 8, 9

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

**Claim 6 — TRUE.** `docs/protocols/AB-PROTOCOL.md:113` is queue item 2
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
whole repo (including `crates/`): hits at `dormouse-spectral/src/lib.rs:225` (doc
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
`vendor/dormouse-fused/crates/dormouse-muon-plus/README.md:30`. Likewise the retracted
`96×/84×`/`3.6×` numbers are in that same **crate** README (`:61-62`, `:75-78`), not
in the project `README.md`'s "Performance" section — which is the two-row
`benches/history.tsv` canary table. `muon-plus.md:183` and `:184` both attribute to
`README.md` and `D15` even names the section, which a reader will resolve to the
wrong file. The *verdicts* survive; the *citations* do not.

---

### 5. Claim 7 — split verdict, with the dependency half **refuted**

**TRUE:** `dormouse-sct/src/qr.rs` does implement the paper's `sign(diag(R))` continuity
correction (`:18-19` names it, `:300-301` and `:351-352` implement it), and
`crates/dormouse-core/src/param.rs:1` does read
`//! param - TSCT linear via dormouse-sct SpectralLinear` while `param.rs:9` imports
`dormouse_spectral::SpectralLinear`. T11 is a correct BUG.

**FALSE: the "dead dependency" (T10).** `dormouse-spectral/Cargo.toml:33` is:

```
27: [dev-dependencies]
...
33: dormouse-sct = { path = "../dormouse-sct" }
```

It is a **`[dev-dependencies]`** entry, not a `[dependencies]` entry — the section
opens at `:27` and the `[dependencies]` block ends at `:25`. Three consequences the
report misses because its grep was scoped to `dormouse-spectral/src/**/*.rs` and so never
saw `examples/`:

1. **It does not drag `dormouse-sct` into any downstream build graph.** A dev-dependency
   is not built for anything that depends on `dormouse-spectral`. The report's
   "**removable today, zero code change**" and "it also drags a second, divergent
   retraction implementation into the build graph" are both false statements about
   how Cargo resolves dev-dependencies.
2. **It is not unused.** `dormouse-spectral/examples/tsct_diag.rs:61-62, 83-88, 105-110`
   constructs `dormouse_sct::SctLinear` and `dormouse_sct::SctConfig` and runs them against
   `SpectralLinear` in a mini-GPT ("does TSCT actually learn? … dense vs
   SpectralLinear vs SpectralMoE, same params budget"). That is a **live, working
   consumer** of the exact-Stiefel oracle.
3. **It is already the shape the report recommends.** `tsct.md:344-347` says "keep
   `dormouse-sct` as a *test-only* oracle and stop linking it into `dormouse-spectral`". A
   dev-dependency consumed by an example **is** a test-only oracle. T10 is
   flagging the recommended state as a defect.

**T10 should be deleted, not downgraded.** It is a BUG verdict on a correct,
deliberate arrangement, which is the failure mode you asked me to hunt.

---

### 6. Delta spot-check

Counts as they actually stand in the report: **Muon+ 4 BUG / 5 DELIBERATE / 8
BENIGN (17 rows)**; **TSCT 3 BUG / 5 DELIBERATE / 2 BENIGN / 3 OURS / 1
UNVERIFIABLE (14 rows)**. (The task brief's "4 BUG / 4 DELIBERATE for TSCT" does not
match the file; the file's own TL;DR says "**One** real BUG" while its table marks
**three**. That internal inconsistency is itself a small defect — the summary
under-reports its own findings by 2x.)

#### BUG verdicts checked (6 of 7)

| id | verdict | my result |
|---|---|---|
| **D1** | BUG | **CONFIRMED**, proved in §1. Strongest finding in either report. |
| **D2** | BUG | **CONFIRMED** in direction; report's ratio 1.48x is **wrong** (1.4545x). Also missed that the comment's "one fewer matmul launch" is false and that the report's own suggested fix inverts toward the slower branch. |
| **D5 / T1** | BUG | **VERDICT CONFIRMED, EVIDENCE REFUTED** — see §3. The paper prints `(1.875, −1.25, 0.375)` in v3 App. D.3. |
| **D14** | BUG | **CONFIRMED** in substance (v3-only, steps-to-target-loss) but the cited file is the **crate** README, not the project README. |
| **T10** | BUG | **REFUTED** — dev-dependency, used by `examples/tsct_diag.rs`. See §5. |
| **T11** | BUG | **CONFIRMED** — `param.rs:1` vs `param.rs:9`. |
| *(not in report)* | — | **MISSED BUG** — see §7.1: 112 unguarded host syncs/step. |

#### DELIBERATE verdicts checked (7)

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

### 7. What the report missed

#### 7.1 The retraction does 112 unguarded host synchronisations per training step — and the report found the code and did not count it

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
- `SpectralLinear::retract` (`dormouse-spectral/src/lib.rs:612-619`) retracts **u and v**.
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
  (`dormouse-spectral/src/lib.rs:275-292`) groups factors by shape, stacks to
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

#### 7.2 The power iteration's start vector is structured, not random — a silent divergence path

Covered under T3 in §6. Stated separately because the report's T3 verdict actively
*credits* the mechanism for robustness it does not have. `g.sum_dim(1)` (`:183`) is
`g·1 = m·(mᵀ1)`; its component along the top left-singular vector of `m` vanishes
exactly when the dominant direction of the factor is orthogonal to the all-ones
vector. Then σ_max is under-estimated, the 1.05 safety factor protects nothing, and
NS diverges. There is no assertion, no fallback, no counter. The honest fix is one
line — a random or deterministic-but-generic start vector, or a Rayleigh-quotient
sanity check — and the honest classification is ADR-0019 **SILENT**, not "the
divergence case is documented".

#### 7.3 The crate README documents a formula the code does not implement

`dormouse-muon-plus/README.md:21` prints the paper's update rule:

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

#### 7.4 The 3.6x claim exists in two places, and the report flagged one

`dormouse-muon-plus/README.md:75-78` repeats the factored-branch claim in prose
("claimed 3.6× faster on [8192,512] with a lower peak footprint (never materializes
(XXᵀ)²)") *after* retracting it 7 lines earlier. D2 flags only `lib.rs:142-145`. The
"never materializes `(XXᵀ)²`" half is a real (and correct) memory argument that
survives the retraction — it is the only true part of the claim — so the fix is not
"delete", it is "keep the memory argument, drop the speed number, in both files".

#### 7.5 The `~40 s/step` that decides the entire routing policy has no provenance

`routing.rs:78-80`: *"8 NS iters on the `[d,d]` projections added ~40 s/step, while
the factored `[d,r]`/`[r,f]` form is ~1000x cheaper"*, echoed in `AGENTS.md` §2.3.
D11 accepts it as "a machine fact". Under **ADR-0020 rule 1** — the rule this very
report is enforcing against `dormouse-spectral` — a measurement is a measurement only
with **config, date and commit**. This one has none, and it is the sole justification
for keeping Muon+ off every `[d,d]` projection, i.e. it decides the optimizer policy
for the whole model. It is also stated **at 8 iterations**, so it is entangled with
D4: change `ns_steps` to the paper's 5 and the number behind the policy is stale.
The report should have held `dormouse-spectral` and `routing.rs` to the same standard.

---

### 8. My top three

1. **§3 — the report's central evidence for its most severe *TSCT* finding is false,
   and its fix makes provenance worse.** Muon+ v3 App. D.1 and D.3 print
   `(3.4445, −4.7750, 2.0315)` and a PolarExpress schedule terminating at
   `(1.875, −1.25, 0.375)`. `dormouse-spectral` uses exactly that triple. The bug is a
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
3. **§5 — T10 is a false BUG.** `dormouse-spectral/Cargo.toml:33` is a
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

### 9. Settling commands for what I could not execute (no GPU, no build per the rules)

| finding | settling command |
|---|---|
| 112 syncs/step, and their step-time cost | `./target/release/train --preset small --batch 8 --seq-len 512 --retract-every 1000` vs default, `--timers`, warm steps 50/100/150, quiet card, `CUBECL_AUTOTUNE_LEVEL=3` |
| `retract_batched` ≡ per-factor path at production shapes | `cargo test -p dormouse-spectral --features cuda ns` (the `retract_batched_deterministic` test already exists at `:1371`; extend it to `[768,64]`, `[2048,64]`, `[768,256]`) |
| NS-3 vs the exact `safe_qr` oracle (the report's own cheapest unrun check) | one test in `dormouse-spectral/tests/` calling `dormouse_sct::qr::qr_cpu` — the dev-dep already exists, which is the whole point of §5 |
| power-iteration start-vector divergence (§7.2) | construct `m` with a top singular vector orthogonal to `1`; assert `polar_orthogonalize` output is finite |
| `~40 s/step` at 8 iters (§7.5) | `--set use_tsct=false` on one preset, `--timers`, warm steps — this is also AB queue item 2 |

## Reviewer B — Review B — muon-plus.md / tsct.md, engineering-consequence pass

**Reviewer 2 of 2, independent.** Angle: *if the report is true and the project follows it, what
actually gets better.* Every citation in both reports is assumed correct — I did not re-fetch
arXiv, did not check the reference implementations, and make no claim about provenance. I
re-derived the numbers instead.

**Not done:** no GPU, no `cargo build`, no `cargo test`. No file other than this one was
touched. `docs/papers/*.md`, the vendor crates and the tree are as the author left them.

**Read:** `docs/papers/muon-plus.md`, `docs/papers/tsct.md`,
`vendor/dormouse-fused/crates/dormouse-spectral/src/lib.rs`, `vendor/dormouse-fused/crates/dormouse-muon-plus/`,
`crates/dormouse-core/src/{param,loop_block,routing}.rs`, `crates/dormouse-train/src/{lib,optim}.rs`,
`docs/protocols/AB-PROTOCOL.md`, `benches/history.tsv`.

**Assumptions I am judging against, stated so they can be rejected:**

- A1. The 246 ms warm step (batch 8 × 512, depth 2, fp32, 9 195 854 params, aux off, `--no-engram`,
  `benches/history.tsv:70-96`) is the live operating point, and the 52.8 ms `retr` inside it is
  the real per-step retraction cost at `retract_every=1`.
- A2. The card is launch-bound (13.3% mean utilisation, 79% of samples ≤5%) and will stay that
  way for the 9.2M-parameter regime.
- A3. `retract_iters = 3`, `retract_every = 1` are the defaults the A/B queue will use
  (`train/src/lib.rs:160`).
- A4. Attention currently receives no gradient, so the 246 ms figure describes a step cheaper
  than a correct one. I price the A/B both ways in §3.

---

### 0. Verdict up front

The reports are **good documents that recommend the wrong actions in the order that matters.**
Both name the TSCT retraction as an unverified 22%-of-step fixed cost. Neither asks what that
cost is *made of*. When I count, the single largest removable item is not the batching the
`tsct.md` §4.4 recommends, and it is not even in the report's findings list: it is a σ_max
**power iteration** that the report §3 T3 calls "our own contribution" and "the reason NS is
viable here at all", and which is **provably redundant** against the Frobenius prescale the
optimizer already uses 130 lines away in the same vendor workspace.

Second, the recommendation the reports do make — wire up `retract_batched` — is **not safe to
follow as written.** Wiring it naively silently freezes every TSCT master.

---

### 1. `retract_batched`: what the syncs and launches actually cost

#### 1.1 The shape and count of the work

`LoopBlock::retract_tsct` (`loop_block.rs:170-176`) walks `expert_ffns[].{gate_up,down}` plus
`out_proj`. `configs/small.toml` is `d_model=768, d_ffn=2048, n_experts=3, rank=64`, so each
`LinearLike` carries a `u:[in,k]` and a `v:[out,k]` (`param.rs:45-58`, both `768` and `2048`
already multiples of 4, so the cubek pad at `param.rs:46-50` is a no-op):

| factor | shape | count |
|---|---|---|
| `gate_up.u`, `down.v` | `[768,64]` | 6 |
| `gate_up.v`, `down.u` | `[2048,64]` | 6 |
| `out_proj.u`, `out_proj.v` | `[768,64]` | 2 |
| **total** | | **14** |

Every factor is tall, so `polar_orthogonalize` transposes to `[64, 768]`/`[64, 2048]` and the
NS works on the small side. `retract_every=1` ⇒ all 14, every step, forever.

#### 1.2 Sync count — the report's "7" is right, and 5 of the 7 are in one loop

`polar_orthogonalize` (`dormouse-spectral/src/lib.rs:168-217`) has exactly three `into_scalar`
sites, and one is inside the power loop:

| line | site | iterations | syncs |
|---|---|---|---|
| `:186` | `v.div_scalar(vn.into_scalar::<f32>())` | `POWER_ITERS = 5` | **5** |
| `:197` | `vgv = v.mul(gv).sum().into_scalar::<f32>()` | 1 | 1 |
| `:198` | `vv = v.mul(v).sum().into_scalar::<f32>()` | 1 | 1 |
| | | | **7** |

**7 per factor × 14 factors = 98 blocking device→host round-trips per training step, every
step, with no counter anywhere.** For contrast, `dormouse-muon-plus` has **zero** `into_scalar` in
its entire hot path (`grep` over `lib.rs` + `fused_kernels.rs`: the only hits are inside
`#[test]` bodies at `fused_kernels.rs:228,361,372,383`). The optimizer runs the *same*
Newton-Schulz iteration on the *same 14 tensors* and is sync-free, because it normalizes with
a `[1]`-shaped tensor and broadcasts (`:137-138`). The retraction does the identical estimate
and reads it to the host five times.

#### 1.3 Launch count — my own accounting

Counting `clone`/`mul_scalar`/`add`/`sqrt`/`clamp_min`/`sum` as one launch each and
`swap_dims` as one transpose copy:

| stage | launches |
|---|---|
| `swap_dims` canonical transpose (`:171-173`) | 1 |
| `g = m·mᵀ` (transpose + gemm, `:182`) | 2 |
| `v = g.sum_dim(1)` (`:183`) | 1 |
| power loop × 5 — `mul`,`sum`,`sqrt`,`clamp_min`,`div_scalar`,`gemm` (`:184-191`) | 30 |
| `gv = g·v` (`:193-196`) | 1 |
| `vgv` (`:197`) | 2 |
| `vv` (`:198`) | 2 |
| `m /= σ·1.05` (`:202`) | 1 |
| NS × 3 — `xx`(2) + `xx2`(1) + `poly`(3) + `combine`(3) (`:206-211`) | 27 |
| un-transpose (`:212-215`) | 1 |
| **per factor** | **68** |
| **per step (14 factors)** | **≈ 950** |

#### 1.4 The batched path, counted the same way

`retract_batched` (`:275-292`) groups by exact shape, so the 14 factors collapse to **two
groups: `[768,64]×8` and `[2048,64]×6`** — no padding, exactly the property the doc comment
advertises. Per group the op count is the same as one scalar factor (same 30-launch power loop,
same 27-launch NS loop) but each op is a batch kernel over 8 or 6 slices:

| | scalar path | batched path |
|---|---|---|
| groups | 14 | **2** |
| launches / step | ≈ 950 | 2×68 = 136, + stack 2, + writeback 14×3 = 42 → **≈ 180** |
| syncs / step | **98** | **0** |

So batching is **5.3× fewer launches and all 98 syncs gone.** The report (`muon-plus.md`
§"batched vs loop") cites 4.1–4.8×, which is the right order for the launch half.

#### 1.5 Cost model, calibrated against a number that is on record

`history.tsv` gives `opt = 43-47 ms` for the *same 14 tensors* (`routing.rs:96` sends
`(Expert|Readout, Factor)` to Muon; `(Head, Factor)` and every `Scale`/`DenseWeight` go to
`Rest`, so the Muon group is exactly the 14 retraction factors). Per tensor that is
8 NS iterations × 9 launches + norm 4 + div 1 + momentum 3 + ColRow 6 + update 3 ≈ **89
launches**, × 14 = **1246 launches in 43 ms ⇒ 34.5 µs/launch.**

That is one parameter fitted to one measurement, so it is a model, not a measurement. It has
one free check: the same model predicts the retraction at 950 × 34.5 µs = 32.8 ms of enqueue
plus 98 syncs. Solving the residual, 52.8 − 32.8 = 20 ms ⇒ **≈ 0.20 ms per sync**. That is a
plausible full queue drain on a launch-bound workload and it lands within 1% of the recorded
52.8 ms, so I am willing to predict with it.

| variant | launches/step | syncs | predicted `retr` | saving |
|---|---|---|---|---|
| **A. today** (scalar + power iteration) | 950 | 98 | **52.8 ms** (recorded) | — |
| **B. batched only** | 180 | 0 | **6–9 ms** | ~45 ms (85%) |
| **C. Frobenius prescale only** (§1.7) | 476 | 0 | **16–18 ms** | ~35 ms (66%) |
| **D. C + B** | 115 | 0 | **5–7 ms** | ~46 ms (88%) |

Three consequences, and the second is the one the reports miss:

1. **Batching is worth roughly 85% of the 52.8 ms and the step drops 246 → ~200 ms (−19%).**
   It is a fixed per-step tax, not a scaling cost — 8× the data moved it 52.8 → 64.6 ms
   (`history.tsv:94`), so it pays identically at 2k steps and at 100k. A 2k-step A/B run goes
   8.2 min → 6.7 min *before* the arm is even considered.
2. **"Remove 7 syncs" and "remove the launches" are not the same win, and the sync story
   understates it.** Variant C removes 100% of the syncs for a 66% saving; B removes the same
   syncs *and* 81% of the launches for 85%. Syncs are worth ~20 ms; launches are worth ~33 ms.
   Anyone who reads `tsct.md` §4.4 ("the claimed sync saving is not being collected") will
   size the prize at 20 ms and under-buy it by a factor of two.
3. **On a 9.2M-param model the FLOPs are irrelevant here, so any launch reduction is free.**
   Total NS arithmetic in the retraction: 14 × 3 × 13.1 MFLOP ≈ 0.55 GFLOP ≈ **80 µs** at
   7 TFLOP/s. Zero point eight percent of the 10 ms it costs. Whatever removes launches wins,
   full stop — which sets up §2.

#### 1.6 `retract_batched` is not a drop-in, and following the report's advice as written would freeze the model

This is the finding I would most want acted on, and neither report has it.

`retract_batched(factors: &mut [&mut Tensor<2>], iters)` (`:275`) takes **raw tensors**. It
writes back with `*factors[i] = out.clone().slice(...).reshape(...)` (`:286-289`). It cannot
call `Param::from_mapped_value`, and it therefore cannot honour the invariant that
`polar_retracked` (`:154-161`) exists to enforce. That function's own doc comment (`:137-153`)
says what happens when you do not:

> *"the polar output is a non-leaf (`GradInBackward`), and burn-optim's step re-tracks
> `Requirement::Grad` only, so a stored non-leaf is silently downgraded to an untracked leaf —
> **the master freezes** and the fused op's backward sees a pruned parent."*

`polar_orthogonalize_batched` does **not** `.detach()` — only the scalar path's
`polar_retracked` does (`:156`). So `retract_batched` as written returns a live autodiff graph
of the whole NS iteration, stored into a `Param`. On the live trainer (`Autodiff<Cuda>`,
`BalancedCheckpointing`) that is a **silent master freeze** — the ADR-0011 SILENT class, on the
one mechanism whose entire purpose is keeping masters trainable, and invisible in the log
because the loss curve of a frozen model still descends.

The tests do not catch it, and structurally cannot:

- all four `retract_batched` tests (`:1327,1371,1396` + the identity assertion) run on
  `dev() = Device::ndarray()` (`:1241`) — **no autodiff device, so `is_require_grad` is never
  true and the freeze is unreachable**;
- the *one* tracking test, `retract_stays_tracked` (`:1315-1320`), builds
  `Device::ndarray().autodiff()` and calls `m.retract(3)` — **the scalar path**;
- the `retract_batched` microbench (`:1396-1438`) is also plain ndarray, and its own comment
  (`:1391-1394`) concedes "no device syncs exist here, so this measures pure
  compute/overhead" — which means **the 4.1–4.8× ratio is a CPU-dispatch measurement and can
  say nothing about the CUDA sync saving.** It is a *lower* bound on the CUDA win, not a
  measurement of it. (`muon-plus.md` then uses that ratio as a *correctness* argument — "a
  ratio on ops with identical gradients is consistent with the reference and is not evidence of
  a divergence". That is a category error: a performance number is not evidence about
  mathematics, in either direction.)

**What to do:** `retract_batched` needs `polar_orthogonalize_batched(...).detach()` plus a
mirrored `set_require_grad`, and its signature has to become `&mut [Param<Tensor<2>>]` (or the
stack/consume/`from_mapped_value` dance moves into `retract_tsct`). Add the missing test on
`Device::ndarray().autodiff()` asserting the master stays tracked after a batched retraction —
one test, and it is the gate the report should have recommended instead of the wiring.

#### 1.7 The bigger fish: the σ_max power iteration is provably unnecessary

Both reports treat the power iteration as load-bearing. `tsct.md` T3: *"**our own
contribution to the retraction** and it is the reason NS is viable here at all."* The code
comment at `:174-180` justifies it: *"A Frobenius/sqrt(k) pre-scale does NOT bound sigma_max:
for a square n×n Gaussian matrix ‖X‖_F ≈ n but sigma_max ≈ 2√n (Bai-Yin), so sigma_max ≈ 2
stays above the basin and NS diverges (measured: polar([512,512], 3) -> max entry ~1e14)."*

Three things wrong with that, in increasing order of consequence:

1. **The code does not do a `‖X‖_F/√k` prescale.** `:202` divides by `σ·1.05` from the power
   iteration; the *Muon* path at `dormouse-muon-plus/src/lib.rs:137` divides by plain `‖X‖_F`. The
   comment is defending against a prescale that is not in the code. The measured 1e14 is a
   historical bug whose trigger is absent.
2. **Plain Frobenius is not just adequate, it is sufficient by Cauchy–Schwarz.** After
   dividing by `‖X‖_F`, `σ_max/‖X‖_F ≤ 1` — *always*, for any matrix, with no distributional
   assumption at all. Bai–Yin is about the un-normalised matrix and is irrelevant.
3. **The cubic's basin is exactly [0, 1], and Frobenius lands inside it.** For
   `p(s) = 1.875s − 1.25s³ + 0.375s⁵`:

   - `p′(s) = 1.875(s²−1)² ≥ 0` — `p` is monotone increasing on `[0,∞)` (verified
     numerically at s = 0, 0.5, 1, 1.4, 2);
   - `p(s) − s = 0.375·s(s²−1)(s²−7/3) > 0` for `s ∈ (0,1)` (verified at s = 0.1, 0.5, 0.9,
     0.99), and `= 0` at `s = 1`;
   - so `p` maps `[0,1] → [0,1]`, strictly increasing, with a **double root at `s = 1`**, i.e.
     cubic-order convergence to the manifold and no overshoot.

   Given `σ_max ≤ 1` after Frobenius normalisation, every singular value is driven
   monotonically to 1 and none escapes. **The iteration converges, unconditionally, with no
   power iteration and therefore no host read of the scale factor.** (The basin bound is 1, not
   the `σ < √3` the comment states — √3 is where the *cubic* blows up, `p(1.5)=1.44`,
   `p(2)=5.75`, `p(3)=63`. Frobenius gives 1. So does the comment's `1.05` safety factor earn
   anything? No: it makes `σ_max·1.05` slightly *worse*.)

**Consequences.** Replacing `:178-201` (5 sequential `[64,64]` gemms + a Rayleigh quotient, 36
launches and **5 of the 7 syncs** per factor) with the 4-launch `‖·‖_F` prescale the optimizer
already uses:

- per factor 68 → **34 launches**, 7 → **0 syncs** (variant C above: 52.8 → ~17 ms);
- it makes the retraction and the optimizer **the same algorithm on the same tensors**, which
  is what §4 of this review shows the owner requires;
- it deletes the thing `tsct.md` ranks as *our own contribution*;
- and it is a ~5-line diff inside `polar_orthogonalize`, with **no change to any call site** —
  so unlike `retract_batched` (§1.6) it cannot break the `Param`/tracking path, because
  `polar_retracked` still wraps it.

**Falsifier, and it is cheap:** on CPU, compare `ortho_error` after 3 cubic steps with a
Frobenius prescale against the same with the power-iteration prescale, at `[768,64]` and
`[2048,64]`, against `dormouse_sct`'s exact `safe_qr`. If the Frobenius variant's residual is
materially worse, restore the power iteration and **add a counter for it** (ADR-0011: an
unmeasured improvement in a hot path is indistinguishable from a no-op). Runs in seconds on
`Device::ndarray()`, needs no GPU, and discharges `tsct.md` §4.3 at the same time.

**Ordering:** do C first, then B. C is 4.6× smaller, cannot break anything, and takes the
saving from 85% to 88% of the ceiling by handing B a shorter per-factor op list to batch.

---

### 2. The dead branch: delete it, but the report's "fix" and the report's reason are both wrong

`muon-plus.md` D1 is correct that `nc * 4 < nr` at `dormouse-muon-plus/src/lib.rs:146` is
unsatisfiable — `nr`/`nc` are read at `:141` from `x` *after* the canonical transpose at
`:130-134`, so `nr ≤ nc` always, and `4nc < nr ≤ nc` has no solution. I confirm it. `D1` also
correctly identifies that the fused `ns_combine_cuda` call at `:154` is consequently dead in
production (I grepped the whole tree: `:154` is its only non-test call site; the test uses are
`fused_kernels.rs:272,331,358`).

**But the report then offers two fixes as equals** — *"either delete the branch or invert to
`nc * 4 > nr` and re-measure"* — and they are not equals. The FLOP count settles it, and the
FLOP count is not what the report thinks it is.

**The comment's arithmetic describes a different algorithm from the code's.** The comment at
`:142-145` compares factored as *"two `[c,c]@[c,r]` matmuls"* against direct as *"`[c,c]@[c,c]` +
`[c,c]@[c,r]`"*. In the code, `xx = x·xᵀ` is `[nr,nr]` — the *small* side — never `[c,c]`. The
comment's "direct" cost (`2c³`) is not a term that exists anywhere in the function.

**The correct forms, for the code as written, `x` being `[nr,nc]` with `nr ≤ nc`:**

```
direct   : xx = x xᵀ [nr,nr]        2·nr²·nc
           xx2 = xx·xx  [nr,nr]     2·nr³
           poly·x       [nr,nr]·[nr,nc]  2·nr²·nc
           ─────────────────────────────  4·nr²·nc + 2·nr³
factored : xx, t1 = xx·x, t2 = xx·t1    6·nr²·nc
ratio factored/direct = 3nc / (2nc + nr)  ≥ 1,  → 1.5 as nr/nc → 0
```

| `nr,nc` | direct | factored | ratio |
|---|---|---|---|
| 64, 768 (**our `[768,64]` factors**) | 13.1 M | 18.9 M | **1.44** |
| 64, 2048 (**our `[2048,64]` factors**) | 34.1 M | 50.3 M | **1.48** |
| 512, 8192 (the comment's case) | 8858 M | 12885 M | **1.46** |
| 768, 768 (square) | 2718 M | 2718 M | 1.00 |

So D2's *conclusion* is right — factored is more FLOPs, and always has been — but its
*stated* arithmetic (`factored = 3·(2nr²nc); direct = 2nr²nc + nr³`) drops a factor of 2 from
both terms of the direct form, and those stated numbers give **2.909** at `[512,8192]`, not
the **1.48** the same cell claims. The cell is internally inconsistent. It does not change the
verdict, and a reviewer checking the FLOPs would have to redo them.

**Now the deletion test, which is where the reports are quiet.**

- *Complexity that disappears:* the 24-line `if` branch (`:147-170`), the `ns_combine_cuda`
  call site, and a comment that is false in two independent ways (wrong Gram shape, and a 3.6×
  figure that `README.md` already retracts for lack of a device flush).
- *Complexity that reappears:* nothing. Today.

So delete it. **But hold on before deleting the kernel.** Count launches again (§1.5):
direct is **9** per NS iteration (`xx` 2 + `xx2` 1 + `poly` 3 + `combine` 3); factored with
`ns_combine_cuda` is **5** (`xx` 2 + `t1` 1 + `t2` 1 + fused combine 1). Over 14 tensors ×
8 iterations = 112 NS iterations per step, that is 1008 → 560 launches, a saving of **448
launches ≈ 15 ms** of a 43 ms `opt` — against **+90 µs** of extra FLOPs (0.14 GFLOP → 0.20
GFLOP at 7 TFLOP/s). On this card the factored+fused form wins by a factor of ~150.

**So the dead branch is not dead code that should be deleted. It is an unexploited lever
behind a wrong condition, and the condition is wrong because it encodes a shape heuristic
where the real decision is a machine property.** The correct gate is not `nc*4 < nr` (a
geometry proxy) and not its inverse (a geometry proxy for the opposite regime); it is *"is this
device launch-bound?"* — which `history.tsv` answers for the current box and which will answer
differently the day dormouse runs somewhere compute-bound. A condition that gives the right
answer on this card and the wrong answer on a H100 is a latent bug, which is exactly the
"3.6× measured on [8192,512]" claim's failure mode all over again.

**Recommendation, in the smallest form that is actually honest:**

1. Delete `lib.rs:146-170` and the `:142-145` comment. One clean commit. This is correct
   regardless of what follows, and it removes a false claim from the tree today.
2. Keep `ns_combine_cuda` in `fused_kernels.rs` — it is correct, tested against the tensor
   path, and now has no production call site (which is ADR-0019's own situation; it needs a
   counter if it is ever wired).
3. Re-add the factored form behind **one `bool` on `MuonPlusConfig`**, defaulted to `false`,
   surfaced through the existing `--set` seam. No new CLI surface. Run it as an A/B with a
   device flush, and record the row in `history.tsv` — that is the measurement whose absence
   produced the retracted 3.6× in the first place.
4. Do **not** flip the condition to `nc*4 > nr`. That would light up 100% of the NS traffic on
   this model, where the 1.44–1.48× FLOP penalty is irrelevant *and* the launch saving is real,
   so it might well win — but it would win for the wrong reason, and the next person to port
   this to a compute-bound card would inherit a silent 1.5× regression with a comment claiming
   a 3.6× *speedup*.

---

### 3. The minimum experiment that would decide TSCT, and what it costs

`tsct.md` §4.1 and `docs/protocols/AB-PROTOCOL.md:113` both say the TSCT-vs-dense A/B is unrun. I think
both name the wrong *first* experiment. In cost order:

#### E1 — does 3 cubic steps actually reach the manifold at our shapes? (CPU, < 5 min, 0 GPU)

`tsct.md` §4.3 says this is the cheapest unrun check. I agree and would put it first,
extended to pin **both** prescale variants (§1.7) against `dormouse_sct`'s exact `safe_qr`:

```
[768,64], [2048,64] × {Frobenius prescale, power-iteration prescale} × 3 iters
  → ‖UᵀU − I‖_F / k   vs   ‖safe_qr(U)ᵀsafe_qr(U) − I‖_F / k
```

This decides whether the retraction is a retraction, and it decides §1.7 at the same time. It
costs nothing and it gates everything below: **if 3 cubic steps do not reach the manifold at
`[768,64]`, the retraction is not a retraction, and the whole A/B measures three arbitrary
programs.** Note the reference already exists in-tree (`dormouse-sct/qr.rs:509`), so this is a
test, not a project.

#### E2 — is the existing ortho test measuring a retraction or a function? (CPU, < 5 min, 0 GPU)

`tsct_retract_restores_ortho` (`train/src/lib.rs:2026-2041`) scales `U` by 3.0 and asserts the
error falls. It proves the function maps a perturbed input somewhere orthonormal. It says
nothing about whether a *real optimizer step* leaves the manifold. Restructuring it to
"one real `optim.step`, then `max_ortho` before and after" is the same cost and is the
question the latch at `train/src/lib.rs:1331-1337` is actually relying on.

#### E3 — do the factors drift at all? (GPU, ~3 min)

The retraction is 22% of a step. The project rule (§1.2) is A/B-or-death, and the *first*
question is whether there is anything to die of. `max_ortho` is checked every 500 steps; at
2k steps that is 4 points, which cannot distinguish a flat 6e-5 from a slow climb to 1e-3.

- **Cost:** one 100-step run with a controlled perturbation of known magnitude injected at
  step 50; assert the 500-step-cadence metric *moves*. If the monitor cannot see a 10× drift
  in 50 steps, the 1e-3 latch cannot see drift either, and the whole `T13` design (which the
  report calls "stronger than the paper's") is a latch that cannot trip.
- **100 steps × 246 ms = 25 s** of GPU. Call it 3 minutes with startup.
- **This is the highest information-per-second experiment on the list** and it needs no
  control, no arm, and no seed spread. If `max_ortho` sits at its 6e-5 floor for 2k steps, the
  answer to the A/B question is "`--retract-every 1000`", which `history.tsv:94` already
  measured as `retr=0.0` and a **188 ms** step against 240 — a free 22% that needs no
  experiment at all, only the evidence to justify it.

#### E4 — the confound-free decomposition (GPU, ~1.5 h at the measured step time)

**The queue's arm 2 cannot answer the question it asks.** `--set use_tsct=false` changes at
least three things at once:

1. the parameterisation (`SpectralLinear` → `burn::nn::Linear`);
2. the retraction (53 ms of the step, and the *only* thing arm 2's own justification
   — "do the TSCT factors, the polar retraction and the quant machinery earn ~1000 lines" —
   names);
3. **the optimizer.** `routing.rs:96` sends `(Expert|Readout, Factor) → Muon`;
   `routing.rs:99` sends `(Expert|Readout, DenseWeight|DenseBias) → Rest`. So the dense arm
   runs **AdamW on the FFN weights** and the spectral arm runs **Muon+ on the factors**. The
   paper's own Table 1 puts Muon+ vs Muon at −0.41 to −2.02 loss — a *known, large, published*
   effect. The arm cannot separate a known large effect from an unknown speculative one.

`docs/protocols/AB-PROTOCOL.md:113` is aware of the parameter-budget confound ("a narrower FFN at the
same param budget") and **says nothing about the optimizer confound**, which is the larger of
the two.

The fix is cheap because the flag already exists:

| arm | flag | parameterisation | optimizer on the FFN | retracts? |
|---|---|---|---|---|
| **A** (control) | — | TSCT | Muon+ on factors | yes, 53 ms |
| **B** | `--factors-fallback` | TSCT | **AdamW on expert factors** (readout still Muon+) | yes |
| **C** | `--set use_tsct=false` | dense | AdamW on weights | no |

A vs B isolates the optimizer routing. A vs C is the protocol's arm. All three at **one batch
size** (else the eval window differs and §2.6 says the numbers are not comparable — batch 2
scores 20 480 B, batch 10 scores 102 400 B).

**Cost at the measured step time (246 ms, batch 8, warm):** 2 000 steps = 8.2 min/run.
3 arms × 3 seeds = 9 runs. The six spectral runs pay the 53 ms retraction; the three dense runs
do not, so ≈ 8 × 8.2 + 3 × 6.4 = **85 min ≈ 1.4 h**, plus eval overhead — call it **1.5 h**.
Three arms for 50% more than the protocol's two, and it is the difference between an answer and
a number.

**And the honest gate, which is the actual deliverable of this section:** this cost is
conditional on the 246 ms figure, and §3.2 of `AGENTS.md` says the attention arm has run no
backward in every run on record (`fused kda=3126/0`). The replacement figure — the tensor-op
KDA backward at 25.8 s/step — has **no committed log and no `history.tsv` row**. If it holds,
9 runs × 2 000 steps × 25.8 s = **129 h**, and the experiment is unfundable as specified;
cutting to 1 seed × 500 steps gives 10.7 h and a noise floor the protocol says cannot decide
anything. **So the step time is not bookkeeping for this A/B, it is the A/B.** One warm-step
measurement with a working attention backward, written into `history.tsv`, must precede any
planning of this queue — which is what `AB-PROTOCOL.md:95-101` already says and I am
endorsing rather than adding to.

#### What the minimum is, stated as one sentence

**Run E1 and E3 first — together under ten minutes of wall clock and three minutes of GPU —
because between them they decide whether the mechanism has anything to be A/B'd about; the
1.5 h three-arm GPU experiment is the *follow-up*, and it is worth nothing if E1 says 3 cubic
steps are not a retraction or E3 says the factors never drift.**

---

### 4. The owner's rule: "deviations allowed ONLY if verified bit-for-bit against a reference implementation"

Applied to the reports' `DELIBERATE` verdicts. Assume every citation in both reports is right;
the question is only what the rule permits *given* that.

| verdict | deviation | rule says | why |
|---|---|---|---|
| **D3** | `lr·max(1,m/n)^{1/2}` vs paper's `lr·(m/n)^{1/2}` | **FORBIDDEN** | The reference is Jordan's `muon.py` and the two are not bit-identical for `m < n`. The report calls it "correctly sourced to Jordan" — **sourcing a deviation to a reference is not verifying it against that reference.** It is a one-line formula, so the bit-for-bit check is trivial; it has not been run. |
| **D4** | `ns_steps = 8` vs paper's 5 | **FORBIDDEN** | Reference is the paper: 5, everywhere, explicitly. A different document (the Qwen report) is not a reference *implementation*; it cannot discharge a bit-for-bit rule. Either revert to 5 or check against Qwen's actual code. |
| **D11** | routing is a *subset* of the paper's | **outside the rule's grammar** | The paper's routing is a hyperparameter choice, not a correctness property. There is no reference implementation to be bit-for-bit with about *which* parameters get which optimizer. The 40 s/step measurement is a machine fact and a fair justification. **The owner should exempt routing by name** — otherwise the rule will be argued about every cycle. |
| **T2** | **cubic (15/8,−5/4,3/8) × 3** vs Muon+ quintic × 5 | **FORBIDDEN** *(and see below)* | The two converge to different points by construction — the cubic reaches `UVᵀ`, the quintic deliberately does not. No tolerance, no iteration count, makes them bit-identical. |
| **T3** | **σ_max power iteration + Rayleigh × 1.05** vs the optimizer's Frobenius prescale | **FORBIDDEN, and the reference is in the tree** | `dormouse-muon-plus/src/lib.rs:137` performs plain Frobenius on the same objects, ~130 lines away in the same vendor workspace. Two NS implementations of the same object, in the same crate, disagreeing. This is precisely the failure the owner's rule exists to stop, and it is the one that costs 7/7 syncs and 39/68 launches per factor per step. |
| **T5** | per-entry `‖UᵀU−I‖_F / k` | **PERMITTED** | The metric is ours; no upstream defines it. The normalisation is a documented bug fix against a dated measurement, not a deviation from a source. |
| **T7** | QR-of-random init instead of SVD | **PERMITTED** | There is no dense matrix to take an SVD of. Nothing to deviate from. |
| **T13** | retraction cadence + one-way persisted fp32 latch | **PERMITTED** | Ours, no upstream. And it is the one that still needs E3 to justify its cost. |

**The test case, T2, and what it actually means.** The owner reads "cubic-vs-quintic NS" as a
mathematical deviation and the rule forbids it. But the rule binds deviations **from a cited
source**, and the honest resolution is not to change the math — it is to **delete the citation**
(T1/D5, the reports' joint #1 and #2 severity). The retraction is not a Muon+ component; it
occupies SCT's Eq. (5) slot and is implemented by our own method. Once "Muon+ §1" is off the
comment, the rule has no purchase on T2 — and the burden moves to a *different* rule, "verified
against a reference", whose natural oracle is `dormouse-sct`'s `safe_qr`. That check is a
**tolerance**, not a bit-for-bit test, because an approximate 3-step retraction can never be
bit-for-bit against an exact one.

**So the owner's rule and the two reports disagree about almost everything, and the
disagreement runs the wrong way.** Of six `DELIBERATE` verdicts: three are permitted (T5, T7,
T13 — all cases with *no* cited source to deviate from), three are forbidden (D3, D4, T2/T3),
one (D11) is outside the rule and needs an exemption. **The three the reports wave through as
"deliberate, documented, correctly sourced" are exactly the three the rule forbids, and the
one the reports rank as a severe bug (D1) is a performance claim the rule does not touch at
all.** Documentation and attribution are doing the work the owner's rule assigns to
verification. The fastest way to comply is a small diff in comments and two one-line
behavioural checks (D3, D4) — *plus* §1.7, which is the only forbidden item with a
performance cost attached.

---

### 5. What would still be unverified after every recommendation here is followed

Taking the reports plus §1.7 plus §2 plus §3-E1…E4 at face value, the following remain open,
and I do not think any of them is closed by a comment or a unit test:

1. **Whether the retraction helps training at all.** E1 closes *is-it-a-retraction*; E3 closes
   *is-it-needed*; E4 closes *does-it-win*. Only E4 answers the BPB question, and E4 is
   unfundable at 25.8 s/step. Nothing in this tree will settle it cheaply, and the honest
   position after all of the above is still §1.2's: the mechanism has never beaten its own
   removal.
2. **Whether the factor-quant forward helps.** `tsct.md` §4.6 is right and unaddressed by
   anything I propose. The latch fired on every fresh run before 2026-09-04, so no run in the
   archive ever engaged the post-fix path, and no run has ever measured it against fp32
   factors. The reports' severity rankings do not mention it; on the A/B-or-death rule it is
   the same class of debt as TSCT itself.
3. **Whether the step-time model in §1.5 is right.** It has one free parameter, fitted to
   `opt = 43 ms`, and validated against `retr = 52.8 ms` — two measurements, one number. It
   is good enough to *order* the fixes and it is worthless as a *quote*. A wrong estimate here
   costs nothing; a wrong estimate quoted in `history.tsv` costs a retracted-claim incident.
   Whichever fix is adopted, its `retr`/`opt` numbers need a real `--timers` row at a stated
   step index, and the count of *syncs removed* should get a counter (ADR-0011 — right now a
   fix that silently failed to apply would look exactly like a fix that worked).
4. **Whether the batched path is numerically identical on CUDA.** Its identity test runs on
   `ndarray` (`:1241`). Batched vs per-factor GEMM uses different tile shapes and therefore a
   different reduction order, so a `< 1e-5` max-diff bound verified on CPU says nothing about
   a `[8,64,768]` fp32 GEMM on sm_120. The retraction decides orthonormality, so this is the
   one place where "close enough" is not obviously acceptable.
5. **Whether the whole cost model survives the attention backward landing.** Everything in §1
   is priced against a 246 ms step in which `fused kda=<f>/0`. If the backward lands at
   25.8 s/step, 52.8 ms of retraction is 0.2% of a step, the batching work is a rounding
   error, and the *only* thing in these two reports worth doing is §4's compliance work. The
   precondition is the same one `AB-PROTOCOL.md:95-101` names and nobody has discharged.
6. **The seed determinism gap.** `AGENTS.md` §3.7 records 409 043 differing values between
   two runs under the same seed after `4b42b6d`. Every A/B number in §3 is downstream of that;
   "3 seeds per arm" is still *nearly* implementable, not implementable.

---

### 6. Top-3 findings

1. **The σ_max power iteration (`dormouse-spectral/src/lib.rs:178-201`) is provably unnecessary,
   is the largest single removable cost in the retraction, and is the item the owner's
   bit-for-bit rule forbids.** Frobenius normalisation gives `σ_max ≤ 1` by Cauchy–Schwarz;
   the cubic `(15/8,−5/4,3/8)` has basin exactly `[0,1]` with a double root at 1
   (`p′(s) = 1.875(s²−1)²`, `p(s)−s = 0.375s(s²−1)(s²−7/3) > 0` on `(0,1)`), so it converges
   unconditionally. Removing it is a ~5-line diff that takes the retraction from **7 syncs and
   68 launches per factor to 0 and 34** — 52.8 ms → ~17 ms, ~14% off a warm step — and it
   leaves the `Param`/tracking path untouched, which `retract_batched` does not.
   *(`tsct.md` T3 rates this as "the reason NS is viable here at all". It is not: the code
   does not use the `‖X‖_F/√k` prescale the comment defends against, and the live Muon path
   at `dormouse-muon-plus/src/lib.rs:137` uses plain Frobenius with no problem.)*

2. **Wiring `retract_batched` as the report recommends would silently freeze every TSCT
   master.** It takes `&mut [&mut Tensor<2>]` (`:275`), cannot call
   `Param::from_mapped_value`, and `polar_orthogonalize_batched` never `.detach()`es — so it
   stores a non-leaf where `polar_retracked`'s own doc comment (`:137-153`) says that silently
   freezes the master and prunes the backward. All four batched tests run on `Device::ndarray()`
   (`:1241`) where `is_require_grad` is never true; the single tracking test (`:1315`) tests
   the *scalar* path. The fix needs the signature change, a `.detach()`, and one
   `Device::ndarray().autodiff()` test. When it is right, it is worth the other half of the
   win: 52.8 → ~7 ms, step 246 → ~200 ms.

3. **Queue arm 2 (`--set use_tsct=false`) confounds the mechanism it is meant to judge, and the
   flag that de-confounds it already exists.** `routing.rs:96` puts TSCT factors in Muon+;
   `routing.rs:99` puts dense weights in `Rest`/AdamW. The arm therefore measures *AdamW-dense
   vs Muon+-spectral* — a published −0.41…−2.02 loss effect tangled with a speculative one.
   Adding `--factors-fallback` as a third arm costs ~25 min more than the protocol's own two
   and turns an uninterpretable number into two readable ones. Relatedly: the `nc*4 < nr` dead
   branch should be **deleted** (its comment is false twice over, and `muon-plus.md` D2's
   correcting FLOP count is itself wrong — `2.909` where it reports `1.48`), but the fused
   kernel behind it is worth 448 launches/step (~15 ms of `opt`) against 90 µs of extra FLOPs,
   so it should come back behind a one-`bool` A/B rather than stay deleted — gated on
   *launch-bound-ness*, not on a shape heuristic that gives the wrong answer on the next card.

**Not claimed:** that any of this improves held-out BPB. It improves step time, or it decides
whether BPB is worth measuring. On the project's own rule (§1.2), the mechanism is still
unjudged, and E1/E3/E4 in §3 are the shortest honest route to judging it.
