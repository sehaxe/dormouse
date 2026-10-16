# JEPA *into* dormouse — the wiring map, the KoLeo replacement, and the one arm to run first

**Date:** 2026-10-16. Read-only: no config touched, no code changed, nothing committed.
**A:** given dormouse's exact forward (ONE weight-shared loop block, 2–4 iterations, byte vocab 256, KDA
linear attention, host Engram, `port/src/dormouse/model/model.py:151-189`), *which* JEPA variant and
*which* exact attachment point is best, and which pieces of the burn-era JEPA+KoLeo recipe (`d8062d1`:
6.343 vs 6.425 BPB, 3/3 seeds) to keep, drop or replace.
**Builds on, does not repeat:** `docs/research/2026-10-16-jepa-dspark-alternatives.md` (the incumbent sweep
— read it first; rows 1/1b/3/5 and §4.1 are the baseline this document moves past) and
`docs/research/2026-10-15-1b-tech-verification.md`. **Method:** full repo read (`aux.rs`, its call sites,
`burn-jepa/{lib,mask,losses,predictor,ema}.rs`, `model.rs`, `fb.py`, `config.py`) + 11 arXiv queries / 21
abstract reads, 2025-01→2026-10. Marks: **(A)** abstract-level claim · **(ours)** measured here, file named ·
**(S)** SPECULATION, no source.

## TL;DR
1. **The burn wiring has neither of the two published conditions for latent prediction to beat token
   prediction:** the predictor sees the FULL latent (`aux.rs:225`) and the target is the SAME depth
   (`out_acc`, `model.rs:304-310` + `:430`) on the SAME inputs — no input bottleneck (`2212.07525`, **A**),
   no "target more abstract than the student's own state" (ProteinJEPA `2605.07554`, **A**). What is left is
   **lagged self-distillation**, which `2607.23531` (**A**) predicts will collapse on text.
2. **The incumbent doc's strongest anti-JEPA citation is misread.** `2604.17228` (**A**) is *not* a JEPA-aux
   A/B but conditional-depth-routing **gates**: the 3/3 win came from removing **oracle `util/rank`
   supervision** (its own caveat: which also cut the FLOPs proxy 1.53×→1.07×), and the *JEPA-guided gate G3
   beat the MLP gate G1 in 3/3 seeds*. **Correct the record** — the case against JEPA is only ProteinJEPA's
   random-init `p=0.059` (**A**) + PRISM (**A**, a VLM regime), weaker than it read.
3. **`jepa_aux_loss` bundles two signals under one weight** — `l1 + 0.1·koleo` (`aux.rs:229`) ×
   `jepa_weight` (`model.rs:~645`), against the repo's own rule (`config.py:168-169`: *"the A/B attributes
   its verdict to ONE signal per run"*). **`d8062d1` therefore cannot say whether the 0.082 BPB came from
   the prediction or the regularizer** — the largest open question here, answerable at zero VRAM.
4. **KoLeo is not the published estimator here** (`aux.rs:227-229`): `mean_dim(1)` pools over time, so the
   statistic runs over `batch` rows (10; 4 where the teacher OOMs the batch) instead of DINOv2's thousands
   of patches. Its successor is the **SIGReg / LeJEPA** family (`2511.08544`, **A**, 2025-11-11;
   `2609.35288` λ-JEPA/SACReg, **A**, 2026-09-28, *improves over LeJEPA*), and `lejepa_loss` is **already
   implemented in our own crate** (`burn-jepa/src/losses.rs:33`) and imported nowhere.
5. **The EMA teacher is the piece to drop.** LeJEPA's own claim is *"no stop-gradient, no teacher-student,
   no hyper-parameter schedulers"* (**A**), and `2609.36227` (**A**) gives the reason a regularizer is
   separable from the predictor: *"an isotropy penalty … its partial derivative in the transition weights
   is zero."*
6. **ONE arm first: `CE + w_reg·SIGReg`, teacher-free, predictor-free, mask-free** — the missing control, on
   position-subsampled latents. Only if it ties, the JEPA-proper assembly: **causal next-latent (predict
   stop-grad `h[p+h]` from `h[p]`, same forward, no teacher) + SIGReg** — no 2026 paper does span/char-level
   masked latent prediction on bytes (two queries, 0 hits), and for a *causal* byte LM "mask a span, predict
   it from before+after" is unrealisable (§1d).

## 1. The wiring map — five attachment points, costed in our terms

| # | where the signal attaches | extra forward | extra params | gradient path | published evidence (id, date, type) |
|---|---|---|---|---|---|
| **a** | per-iteration `out_acc` vs EMA teacher of the SAME model (the burn shape) | **yes — a second full forward**; measured: aux off = batch 16 @ 5908 MiB, teacher on ⇒ batch 4 OOMs (**ours**) | **+1× model**, serialized in every checkpoint (`train/src/lib.rs:459-490,563-581`) | L1 → `JepaPredictor` → `out_acc` → all student params; teacher `no_grad`-frozen (`aux.rs:274-282`) | **ours** `d8062d1` 3/3; no ≤100M AR byte-LM study exists either way. Nearest: `2605.07554` 2026-05-08 (**A**/**M**, protein ESM2, `p=0.059` from random init), `2606.05753` 2026-06-04 (**A**/**M**, VLM latents) |
| **b** | the FUTURE latent, stop-grad, **from the same forward** — predict `h[p+h]` from `h[p]`; no teacher | **no** (h[p+h] is already computed in this batch) | +`d_model²` predictor only (or 0 if reused) | loss → predictor → `h[p]` → backbone; target detached. In a causal model the predictor's input has strictly less information than the target's — the bottleneck (a) lacks | technique real: `2609.36227` 2026-09-28 (**A**, time series, its headline is a roll-out negative, silent on LM loss); `2608.05806` 2026-08-06 (**A**/**M**: plain NextLat weak alone); `2607.23531` 2026-07-26 (**A**, T-JEPA on text). Cross-batch caching instead of `+h`: 1 extra `[b,t,d]` f32 ≈ 15.7 MB + a resume-sidecar — strictly worse |
| **c** | against the Engram readout (predict `out_acc` from (hash-keys, hidden)) | no | ~0 | self-supervised by construction | **none.** `2601.07372` 2026-01-12 (**A**) argues *against* spending an aux here: the lookup "relieves the backbone's early layers from static reconstruction" — it already covers the local/static part. **(S)** |
| **d** | sub-byte temporal spans, data2vec-LLM shape | **yes** — a real input bottleneck needs the target's view on the UNMASKED input (`2605.07554`: "half-depth hidden states of a teacher **given the unmasked sequence**") ⇒ a second forward, i.e. (a)'s price | +teacher | as (a) | **no byte/char-level paper found** (2 queries, 0 hits). In a *causal* AR model "predict the masked span from before **and after**" is unrealisable; it degenerates to (b) with a masked prefix. **(S)** |
| **e** | drop JEPA; future-byte head + probe (incumbent verdict) | no | **196 864 = 2.14%** of `small` (**ours**, `fb.py:FB_PARAMS_AT_SMALL`) | CE → head → `norm(out_acc)` → backbone; no teacher, no detach, one plain CE | `2404.19737` 2024-04-30 (**A**: "no overhead in training time", "increasingly useful for larger model sizes"); the on-regime negative is `2505.22757` 2025-05-28 (**A**: SLMs struggle with MTP). Pre-registered: AB-PROTOCOL row 1b |

**Reading of the table.** (a) pays the binding resource (`AGENTS.md` §2.2) for a signal whose own 2026 paper says it works *without improving the loss*; (c) has no evidence; (d) does not exist causally; **(b) is the only latent-prediction shape that costs nothing and satisfies the bottleneck condition**.

## 2. KoLeo specifically, and what 2026 replaced it with
- **What it does in the burn recipe:** a uniformity pressure on the **time-pooled** latent —
  `student_latent.mean_dim(1)` → `[b, d]` → `-mean(log nn_dist)` with a soft-min (τ=0.25) surrogate for the row-min (`aux.rs:227-229`; `losses.rs:63-97`; DINOv2 `2304.07193`, **A**/**M**).
- **The sample count is the defect:** it runs over **`batch` rows** (10; 4 on the arm where the teacher
  OOMs the batch), not the published estimator's thousands of patches; at `n ≤ 1` `koleo_loss` degenerates
  to the constant `-log(1e3) ≈ -6.908` (`losses.rs:66,86-97`, **ours**).
- **The 2025-26 successor is the SIGReg / LeJEPA family.** `2511.08544` 2025-11-11 Balestriero & LeCun
  (**A**): isotropic-Gaussian regularisation, *one* trade-off hyperparameter, linear time/memory, **"no
  stop-gradient, no teacher-student, no hyper-parameter schedulers"**, ~50 lines. Already in our tree:
  `lejepa_loss` = `‖mean(z)‖² + ‖cov(z) − I‖²/D` (`losses.rs:33-50`), **imported nowhere**. VICReg
  (`2105.04906`, 2021) is **not** the successor: `2607.13612` 2026-07-15 (**A**, Lean-4-verified) finds
  *"VICReg and LogDet unsafe upper bounds, PairDist a safe lower bound, and SIGReg eliminates the gap."*
- **2026 split into two directions.** (i) *Better regulariser:* `2609.35288` 2026-09-28 λ-JEPA/SACReg (**A**)
  beats LeJEPA and VISReg on ImageNet-1k and 8-dataset transfer (LeVJEPA/V-JEPA 2 on video); its
  motivation is literally our defect — regularising a projection while the backbone stays rank-starved.
  (ii) *Anti-collapse from the data rather than a prescribed geometry:* `2608.17542` 2026-08-18 AC-MTM
  (**A**, inverse dynamics + NCE; matches SIGReg on average, beats it 80.0±2.0% vs 58.0±2.0% where the
  Gaussian becomes a bottleneck) and `2609.23881` 2026-09-20 MotionJEPA/DISReg (**A**: plain JEPA has a
  *strong inductive bias towards slow features* — **a lagged EMA target is a slow-feature target**).
- **Cheapest anti-collapse IF JEPA stays:** `lejepa_loss` on **[S, d], S ≈ 256 positions subsampled by
  stride across the batch** — 3 tensor ops (mean, cov, eye), O(S·d²), **cheaper than KoLeo's O(S²) pairwise
  distances**. Not `mean_dim(1)`.

## 3. The recipe assembly — the ONE arm to implement first on the port
**Arm "reg-only" (the missing control):** `loss = CE + w_reg · lejepa_loss(z)`, `z` = `norm(out_acc)` (the
state `lm_head` reads, `model.py:162`) reshaped to `[S, d]` by striding the `b·t` positions. **No teacher,
no predictor, no mask, no second forward, no new parameters.** mechanism ← `2511.08544` (**A**) +
`2609.35288` (**A**). `w_reg` ∈ {0.005 (= today's effective `0.05 × 0.1`), 0.05, 0.1} — publish which.
- why first: it isolates the only half of `d8062d1` testable without paying the VRAM price — the burn term
  cannot separate prediction from regularisation (`aux.rs:229`, `model.rs:~645` vs `config.py:168-169`) —
  and `2609.36227` (**A**) licenses the control: the regulariser's gradient into the transition weights is
  **exactly zero**.
- **Pre-registered A/B:** 3 seeds × **2 000 steps**, `preset small` (9.2M), the port's landed WSD recipe,
  **batch × seq_len equal across arms** so the eval window (81 920 B) matches (`AGENTS.md` §2.6); control =
  the port's existing pure CE (BPB **3.001** @5k, `1b-readiness-plan.md:479`). Metrics in order:
  (1) held-out **BPB** at matched steps, window printed; (2) **the void-check counter** — a `probe::SIGREG`
  bumped *where the term is added* and asserted in `preset_exec` (the `probe::JEPA` precedent: a counter
  bumped nowhere made the arm look alive, `model.rs:637-646`); (3) **PRISM** — a linear probe on
  `norm(out_acc)` for the byte at `t+1` plus a **corruption test** (`2606.05753`). Win = mean BPB below the
  CE mean by more than the spread across the control's own 3 seeds; inside the spread = **delete**
  (ADR-0002).
- **Predicted outcome (S): tie ~55-60% · win ≤0.05 BPB ~25% · loss ~20%** — `d8062d1`'s effect (0.082) sits
  at the edge of plausible seed spread, the standalone-regulariser support (`2603.05924`, **A**,
  single-author preprint, supervised vision) is off-regime, and the on-regime random-init study
  (`2605.07554`) replicates at `p=0.059`.
- **If it ties:** arm **"nextlat+SIGReg"** — target `stop_grad(h[p+h])` (`h = 2` bytes to start;
  `2609.36227` favours a *window*: 8-step context reaches 0.056 vs 0.778 at 16 steps), input `h[p]`, loss
  **cosine, not L1** (`2607.23531`, `2605.07554`). **If it wins:** delete the teacher, the predictor and the
  mask from the JEPA lane. Either way: 2 000 steps at 9.2M **falsifies cheaply and cannot confirm**.

## 4. What to delete from the burn recipe, and the D1-style check on `aux.rs`
**Delete (2026 evidence says dead weight):**
1. **The EMA teacher + its live forward** (`aux.rs:274-282`, `model.rs:304-310`): a halved legal batch and
   1× model params for an objective whose own paper needs *no teacher-student* (`2511.08544`, **A**), plus
   **KoLeo on `mean_dim(1)`** (`aux.rs:227-229`) — the wrong estimator, superseded by SIGReg (§2).
2. **The bundled weight** `l1 + 0.1·koleo` under one `jepa_weight` (`aux.rs:229`, `model.rs:~645`): two
   signals, one attribution — forbidden by the repo's own rule (`config.py:168-169`), and the reason
   `d8062d1` cannot be attributed. **Unramped fixed `TEACHER_MOMENTUM = 0.999`** (`aux.rs:25`) goes with it:
   our own crate's doc says a ramp is recommended (`burn-jepa/src/lib.rs:20`).
3. **Keep, with a caveat:** the offline lane `--jepa-precompute`/`--jepa-targets`
   (`train/src/lib.rs:173-176,728-790`) is the one burn-era piece worth keeping — it converts (a)'s
   "husband the batch forever" into "one precompute pass, full batch in the hot loop". Moot once the
   teacher is gone; if (a) is ever re-run for a shape claim, run it *through this lane*.

**`aux.rs` defect list (read line by line; the file is the burn-side formula source):**
- **D-a — the doc is inverted, the code is right.** `:169-171` says *"a start covers `span` positions
  ENDING AT ITSELF"*; `:176-179` masks `span` positions **starting** at the start, and
  `burn_jepa::mask_indices` (`mask.rs:31-35`, `c − shift(c,span) > 0` = "≥1 start in `(i−span, i]`") is
  forward too. The FNV golden (`:885`) pins the bits, so "fixing the prose into the code" would be caught.
- **D-b — the EMA arithmetic has no coverage.** `:259` is `t.mul_scalar(m) + s.mul_scalar(1−m)`, and
  **every call site in the tree passes momentum 0.0** (`future_byte.rs:314`, `jepa_teacher_seam.rs:111,210`,
  `preset_exec.rs:158,609`) — deleting `mul_scalar(self.m)` leaves the suite green. `burn-jepa`'s
  `ema_update_tracks_student` (`lib.rs:86`) tests the *scalar* `EmaTarget`, not this function.
- **D-c — a same-shape param swap is silent.** `ParamCollector` walks the **student** (`:275-276`),
  `EmaMapper` the **teacher** (`:281`); the only guard is `expect(..)` on a **count** (`:256`). Two
  `[768,768]` projections swapped corrupt the teacher silently — the §1.1 defect class.
- **D-d — KoLeo is not the published estimator:** `:227-229` (see §2); at `n ≤ 1` it returns the constant
  `-log(1e3)` (`losses.rs:66,86-97`). **D-e — no input bottleneck and the target is the same depth:**
  `:225` feeds the predictor the **full** latent, and the teacher target is `out_acc` at the same depth on
  the **same** inputs (`model.rs:286-310`).
- **D-f — the wrong latent metric.** `:226` uses L1; three 2026 sources point to cosine/normalised
  (`2607.23531` squared-error mismatch for text; `2605.07554` cosine beats MSE; `2606.05753` cosine
  *alignment* anti-correlates with accuracy at r=−0.94 — as a *metric*, hence the probe).

## 5. New verified ids (fetched 2026-10-16, arXiv API; date = the API's `published`)
| id | date | who | one line |
|---|---|---|---|
| **2511.08544** | 2025-11-11 | Balestriero, LeCun | LeJEPA/SIGReg: isotropic Gaussian, one hyperparameter, *no teacher-student, no stop-gradient, no schedulers*; ViT-H/14 79% IN-1k linear probe. |
| **2607.23531** | 2026-07-26 | Dinh, Vo | *The JEPA Paradox in Language*: three conditions (predictability, non-collapse, **low conditional variance**); matched I-JEPA/T-JEPA on text; collapse order = MI saturation → target-variance rise → rank degeneration → cosine collapse, over 5 seeds. |
| **2609.36227** | 2026-09-28 | Wang, Cai, Hong | LeNEPA: next-latent prediction, the isotropy penalty replacing stop-grad; **the isotropy penalty's derivative in the transition weights is zero**; a window beats one step; its roll-out claim is negative. |
| **2609.35288** | 2026-09-28 | Demirel, Dominé, Maiorca, Fumero, Mondelli, Locatello | λ-JEPA/SACReg: spectral anti-collapse **on the backbone**; beats LeJEPA and VISReg on IN-1k + 8-dataset transfer, LeVJEPA/V-JEPA 2 on video. |
| **2609.23881** | 2026-09-20 | Karmann et al. (Balestriero, Torr, …) | MotionJEPA/DISReg: plain JEPA has a **slow-feature bias** causing collapse; a dynamic term with no constraint on the embedding's shape fixes it. |
| **2608.17542** | 2026-08-18 | Boylan, Hokamp | AC-MTM: anti-collapse from the transition data (inverse-dynamics + NCE), no target net/stop-grad; 80.0±2.0% vs SIGReg 58.0±2.0% on OGBench Visual Scene. |
| **2603.05924** | 2026-03-06 | Akbar | Weak-SIGReg as a general optimiser stabiliser (covariance via random sketching); ViT CIFAR-100 collapse 20.73%→72.02%. Low-credibility venue: preprint, single author. |
| **2605.27734** | 2026-05-26 | Korchinski, Favero, Wyart | Latent prediction needs **constant** samples in latent depth vs exponential for token-level SSL — for a *hierarchical* latent, not a lagged self-target. |
| **2608.28625** | 2026-07-31 | Zuo, de la Clergerie, Sagot | SciJEPA: asymmetric within-document prediction (title+abstract→method→conclusion); plain predictive training loses to contrastive, **SIGReg closes much of the gap**. |
| **2601.07372** | 2026-01-12 | Cheng et al. (DeepSeek) | Engram: conditional memory as a sparsity axis; the lookup frees the backbone's early layers from static reconstruction. |
| **2607.13612** | 2026-07-15 | Arnez, Gomez-Villa | SIGReg as variational free energy: VICReg/LogDet unsafe upper bounds, PairDist safe lower, **SIGReg eliminates the gap**; Lean 4-verified. |
| **2105.04906** | 2021-05-11 | Bardes, Ponce, LeCun | VICReg: variance + invariance + covariance — real, but ranked *unsafe* by `2607.13612`, so not the KoLeo successor. |
| **2604.17228** | 2026-04-19 | Lin | **Correction:** *Revisiting Auxiliary Losses for Conditional Depth Routing* — routing **gates** at 157.5M. The 3/3 win is from removing oracle `util/rank` (confounded with −39% FLOPs); **the JEPA-guided gate G3 beat G1 3/3**. Not a JEPA-aux negative. |
| **2601.16531** | 2026-01-23 | Lin | Engram-Nine (already in `2026-10-15-1b-tech-verification.md`); carried because `aux.rs`'s KoLeo and Engram paths both touch hashed keys. |

## 6. VERIFIED (sourced) vs SPECULATION (my inference)
**VERIFIED.** Every id above resolved through the API on 2026-10-16 at the date shown; numbers are the
authors' abstract-level claims unless marked **(ours)**. Ours, from the tree: the JEPA target is `out_acc`
at the same depth on the same inputs (`model.rs:286-310`, `:430-431`); the predictor is fed the unmasked
latent (`aux.rs:225`); KoLeo runs on `[b,d]` (`aux.rs:227-229`); the term is `l1 + 0.1·koleo` under one
weight (`aux.rs:229`, `model.rs:~645`); every `ema_update` call site passes momentum 0.0; `lejepa_loss`
exists at `losses.rs:33` and is imported nowhere; the port refuses JEPA/DSpark (`config.py:181-182`) and
refuses `aux_fb_weight` with them (`config.py:364-372`); the fb head is 196 864 params, one plain CE, with a
`fb=<ran>/<asked>` counter (`fb.py:44,105-120`, `train/loop.py:47-58`); `2604.17228`'s misreading stands.

**SPECULATION (mine, no source).** **S1** that a **regulariser-only** term is the right *first* arm — the
sizing (3 seeds × 2 000 steps) and the attribution argument are mine, and `2603.05924` is off-regime.
**S2** that a **causal** next-latent target escapes `2607.23531`'s conditional-variance collapse because the
target is a deterministic function of the input rather than one of several valid completions — the paper's
conditions support this reading, the paper does not test it. **S3** the predicted 55-60% tie in §3: an
inference from our own effect size and the control's unknown spread, not a measurement. **S4** that a
corruption test would show the burn JEPA term is not load-bearing (the incumbent doc's S4, unchanged).
**S5** that an inverse-dynamics/NCE anti-collapse (`2608.17542`) fits an enumerable 256-symbol byte vocabulary.
**I cannot answer on available information:** (i) whether **any** 2026 study gives a latent-prediction aux
win on an **AR byte-level or ≤100M** LM at the loss — none exists in either direction; (ii) whether a
**char/byte-level** span-masked latent objective has ever been published — two targeted queries returned
**zero** relevant hits; (iii) the **VRAM** cost of any draft/JEPA head in the published record.

## 7. Open questions
1. Does the **corruption test** show `d8062d1`'s contribution is load-bearing on `out_acc`? One checkpoint,
   zero the aux path, re-score — the cheapest experiment in this document.
2. What is the control's **own 3-seed spread** at 9.2M/2 000 steps? Nothing in §3 can be decided before it
   is printed, and `d8062d1` never printed it.
3. Does **SIGReg-only** reproduce 6.343? That arm decides whether the teacher exists at all. And
   `docs-site/tools/manifest.mjs` needs a line for this document before `npm run check` is green — not done.

## 8. Sources
arXiv, fetched 2026-10-16 via the API: `2105.04906` 2021-05-11 · `2212.07525` (data2vec 2.0, via
`burn-jepa/src/lib.rs:13`) · `2304.07193` (DINOv2/KoLeo, via `losses.rs:15`) · `2404.19737` 2024-04-30 ·
`2505.22757` 2025-05-28 · `2511.08544` 2025-11-11 · `2601.07372` 2026-01-12 · `2603.05924` 2026-03-06 ·
`2604.17228` 2026-04-19 (**corrected here**) · `2605.07554` 2026-05-08 · `2605.27734` 2026-05-26 ·
`2606.05753` 2026-06-04 · `2606.30356` 2026-06-29 · `2607.13612` 2026-07-15 · `2607.23531` 2026-07-26 ·
`2608.05806` 2026-08-06 · `2608.17542` 2026-08-18 · `2608.28625` 2026-07-31 · `2609.23881` 2026-09-20 ·
`2609.35288` 2026-09-28 · `2609.36227` 2026-09-28. Repo: `crates/dormouse-core/src/{aux,model,future_byte}.rs` · `vendor/dormouse-fused/crates/burn-jepa/src/{lib,mask,losses,predictor,ema}.rs` ·
`crates/dormouse-train/src/lib.rs` · `crates/dormouse-core/tests/{jepa_teacher_seam,preset_exec}.rs` ·
`port/src/dormouse/{config.py,model/{model,fb}.py,train/loop.py}` · `benches/history.tsv:51-56` ·
`AGENTS.md` §2.2/§3.2 · `docs/decisions/2026-10-15-1b-readiness-plan.md:477-479`.
