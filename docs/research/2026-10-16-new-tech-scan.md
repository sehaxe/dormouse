# New-tech screening sweep 2026-07..10 — what dormouse has NOT evaluated

**Date:** 2026-10-16. Read-only: this file only; no config or code touched, nothing committed.
**Ask:** a wide screening sweep for technologies NEW in 2026 (priority 2026-07..10) that dormouse has not
evaluated, ranked by expected-value × fit for this exact stack — byte-level AR (vocab 256), weight-shared
loop, KDA + Engram + TSCT MoE, ≤1B on one 16 GB sm_120 card, Rust (`port/`) + burn.
**Method:** 40 arXiv-MCP queries over 8 white-space lanes (2026-06-15..2026-10-16, date-sorted), every
candidate ID then resolved individually with `get_abstract` (never from a title). Filtered against the
**504 distinct IDs already cited** in `docs/research/*.md` (320 of them 26xx); same-topic survivors appear
below only as **VERDICT-CHANGE**. Every ID here was verified real **on 2026-10-16**; the table date is the
paper's first submission. Marks: **[A]** = authors' abstract, quoted · **[ours]** = repo fact, file named ·
**[S1/S2/S3]** = SPECULATION (S1 sourced mechanism, scale-mismatched transfer · S2 topology argument · S3 none).

## 0. Already swept — do not re-screen
Muon + hybrid routing (`2610.17429`, `2509.23106`, `2607.04422`, `2609.11655`) · KDA decay forms + KDA ceiling
(`2026-09-27-kda-sota-ceiling.md`) · recurrence φ / `2604.21106` (`2026-10-11-recurrence-equivalence.md`) ·
MoD/MoR/MoE nulls (`2026-10-16-token-skipping.md`) · JEPA/SIGReg/LeJEPA (`2026-10-16-jepa-*.md`) ·
DSpark/MTP (`docs/papers/dspark.md`) · one-button pipelines · agentic post-training · budget-savers SL1–SL3 ·
TSCT/LittleBit quant ladder · MSA sparse re-entry (recipe fixed) · Nemotron-CC/data strategy.

---

## TL;DR
1. **[A] Our loop is exactly the regime one Oct-2026 paper says is stuck.** `2610.01153` opens: *"the gains
   from additional iterations diminish quickly and can even turn into degradation, so the extra FLOPs spent
   on looping yield little substantial improvement. Consequently, prior work typically settles on two loops."*
   That is `configs/b1.toml` (`max_iter 4`, operating depth 2) and `small` (operating depth 2). Its fix
   (LOOM: bounded residual updates + input re-injection per loop + **per-loop routers** + a looping residual)
   is measured at **100M–1.7B, stable to 9–12 loops** — our scale band, not a toy.
2. **[A] A newer paper changes the recurrence verdict we hold.** Our `φ = 0.46` is a single-axis fit
   (`2026-10-11-recurrence-equivalence.md`). `2609.40316` fits recurrence **jointly with sparsity**:
   *"sparsity delivers ~3x active-parameter efficiency, recurrence yields ~2x total-parameter efficiency on
   reasoning"*; at matched compute a law-derived looped MoE matches a **~2x larger non-looped MoE**.
   **VERDICT-CHANGE** for 2604.21106: the loop's value is *conditional on MoE sparsity*, which our φ fit cannot see.
3. **[A] Newest byte-level result is about WHERE to spend compute, and it is cheap.** `2610.22100` (Oct 15):
   BLT's entropy rule *"rarely starts a patch at in-line computed results, such as the number after "=" in a
   worked solution"*; novelty patching (a 2-byte lookup table, *"fitted in seconds on raw text with no labels"*)
   takes final-answer accuracy **60% → 90% at 50M** and computed-result accuracy **21% → 57%** (3 seeds each).
4. **[A] A hard RISK to b1 that nobody has screened: MoE + repeated data.** `2609.11917` (80M–1B active,
   8.5B total): *"80M dense models can repeat data over 8x with minimal degradation, MoEs instead begin to
   suffer at 4x"*, and *"no method fully matches the performance of all-unique training data"*. Our corpus is
   repeated by construction [ours, `2026-09-26-byte-lm-recipes.md`: D_u = T/4, ≤4 epochs free]. **VERDICT-CHANGE**
   for the b1 MoE and for the MoE null-result file.
5. **[A] The cheapest real adoption is on the arm we already own.** `2609.24797` Complex KDA: gates in
   `[-1,1]` plus `β ∈ [0,2]` *"preserves KDA's stability and efficiency, with transitions that remain
   diagonal-plus-rank-one and non-expansive"*, reaches `DeltaProduct₂` expressivity, and wins
   *"length extrapolation among tested KDA range settings"*. Two parameter-range lines on our KDA arm.
6. **[A] Muon's small-scale budget story is NOT what our plan assumes.** `2610.21555` (17.7M/33.6M on one
   A100): Muon cuts validation AUC 3.25–22.44% at equal data, but *"AdamW is better by 3.38–6.96% on
   language"* under **time-normalized** loss, and the Muon call eats **56.1–69.6% of training time**. We are
   launch-bound with opt at 43–47 ms of a ~245 ms step [ours, memory 2026-09-29] — a different regime, but a
   standing caution against "Muon everywhere".
7. **No-hit items are stated as such (§4):** `MuonClip`, `LoMu`, `PolarGrad`, `delta-product`, `PSGD`
   successors, `schedule-free Muon` (pre-July only) — **no real id found** for any of them. Byte-level
   spec-decode newer than `2610.05978` — **none**; `2610.20503` is hybrid-linear-attention, not byte.

---

## 1. Candidate table (STEP 2) — ranked by expected value × fit

`Verdict` = ADOPT / SKIP / READ-ONLY. `Cost` = LOC-scale + the gate that must pass.

| # | candidate (id, submitted) | core claim [A] | measured scale | OUR fit | verdict + cost | A/B or integration point |
|---|---|---|---|---|---|---|
| 1 | **LOOM** `2610.01153` 2026-10-01 | looped MoE stalls at 2 loops from **hidden-state variance growth** + **expert-selection collapse**; fix = bounded residual updates, input re-injection per loop, per-loop routers, looping residual → stable 9–12 loops | 100M–1.7B | exact: we are `max_iter 4` / depth 2, shared block, topk-2 of 1024 [ours b1.toml] | **ADOPT** — new loop arm: residual scaling + per-loop router; medium LOC, touches the core loop | 3 seeds × 2k pure CE at `small`; read the depth curve + φ, not BPB alone (AB-PROTOCOL §verdict metric) |
| 2 | **Complex KDA** `2609.24797` 2026-09-21 | KDA reaches 2-D rotation with one delta step + its channel gate reflection; needs gates in `[-1,1]` and `β ∈ [0,2]`; stays diagonal+rank-one, non-expansive; every orthogonal DP-1 matrix is exactly a CKDA transition | S₃/S₄, periodic audio; "outperforms Transformers" (lang) | direct: our KDA arm already carries q,k,v,g + β [ours, small.toml param table] | **ADOPT (cheapest)** — 2 range lines + a stability assert; watch FLA-kernel assumptions | `small`, 3 seeds, 2k pure CE, KDA arm identical otherwise; gate = no NaN + BPB ≤ control |
| 3 | **MoE vs repeated data** `2609.11917` 2026-09-10 | MoE degrades at 4× repetition where 80M dense survives 8×; worsens with total-param sparsity; masking-based regularization helps but never matches unique data | 80M–1B active / 8.5B total | our 19.7 GB corpus is re-read ~4× [ours, byte-lm recipes] and b1 is total-1.07B/active-57M | **VERDICT-CHANGE gate** — read before any long b1 run; cheap | re-run the existing data-repeat ladder **with the MoE arm on**; add dropout/mask regularization as the second factor |
| 4 | **Novelty patching** `2610.22100` 2026-10-15 | patch where bytes are unlikely to repeat recent context (LZ77 view), 2-byte lookup table; beats BLT's entropy rule on computed results | 1M / 12M / 50M BLT-style, 1.9 GB math, 3 seeds | byte-level ✓, but we are not patchified today (vocab 256, seq 2048) | **ADOPT-if-patching** — a table + a patcher hook; small LOC, blocked without a BLT arm | patch-rate-matched arm vs the entropy rule at ~11% patches; metric = BPB on our eval + computed-result accuracy |
| 5 | **Loop Scaling Laws** `2609.40316` 2026-09-30 | first law fitting recurrence **and** sparsity with size/data; looped MoE matches ~2× larger non-looped MoE at matched compute; test-time scaling via recurrence | law-fit + trillion-token check | replaces our single-axis φ as a planning instrument | **ADOPT (instrument)** — no training code; a fit over existing runs | refit on our existing ladder + the MoE arm; if it reproduces φ=0.46 as a special case it is validated |
| 6 | **What makes recurrence effective** `2609.36636` 2026-09-29 | recurrence improves reasoning *beyond* the training horizon while **degrading knowledge**; placement matters; non-recurrent output layers are robust to under-unrolling; **history-state injection** beats initial-state injection | controlled, budgets below/at/beyond horizon | we reset `h = h_ctx` per iteration [ours loop_block.rs:381] | **ADOPT (cheap config)** — readout/injection variant behind a flag | `small`, 3 seeds; compare with the `--rand-depth` arm (AB-PROTOCOL row 4) |
| 7 | **LatentLoop** `2610.19975` 2026-10-14 | loop visits play different roles: the **first** visit's residual scale is smaller; shrinking it preserves/improves, shrinking the second hurts → asymmetric allocation | shared block, 2 visits | we have `iter_embed (4×768) + residual_scale` [ours small.toml] | **ADOPT (cheap)** — one coefficient per visit | sweep per-visit residual scale at `small`; 1 seed per rung, 3 total |
| 8 | **Sparse Delta Memory** `2607.07386` 2026-07-08 | replace the dense KV outer product with **sparse reads/writes to a large explicit memory**; isoFLOP + iso-params, higher state capacity improves ICL and long-context retrieval; learned initial state = parametric memory | long-context retrieval + common-knowledge | extends GatedDeltaNet — the family our KDA sits on | **ADOPT as a successor arm** — medium LOC (sparse addressing); real sm_120 gather risk | arm vs KDA at `small`, 3 seeds; gate = same param count, BPB + retrieval probe |
| 9 | **Kalman Delta Networks** `2609.07816` 2026-09-07 | delta-rule write strength is a Kalman gain; keep covariance **anisotropic** and make both noise variances input-dependent | associative recall | KDA sets write strength from the token only, as the paper says | **ADOPT as a KDA variant** — medium LOC, additive to the arm | `small` A/B vs vanilla KDA, 3 seeds, same β/gate ranges |
| 10 | **Distance curriculum** `2609.16183` 2026-09-14 | short causal conv dominates recall (+0.5); rank-1 vs diagonal margin falls to +0.03 once conv present; under sparse supervision a **distance curriculum takes 0.021 → 1.000** on unchanged architecture | matched state budget, L=256/512 | we have no short conv on the KDA arm; our corpus is domain-sorted | **ADOPT (schedule only)** — cheapest item in the table | add a distance curriculum to the existing arm; also test a 3-4 wide causal conv in front of KDA |
| 11 | **MirrorMoE** `2610.22353` 2026-10-15 | shared experts **reduce routing exploration**, which conflicts with fine-grained segmentation; token-adaptive suppression of unnecessary experts reconciles it | "across model scales" | we have an always-on shared expert (rank 192) next to 1024 routed experts, topk 2 [ours b1.toml] | **ADOPT (probe first)** — measure router entropy across loops before coding | instrument: per-loop expert-selection overlap; if collapse exists, this is the fix |
| 12 | **k1/k2 decoupling** `2609.04575` 2026-09-04 | top-k renormalization calibrates expert gain to the training k; activating k1 while normalizing by mass k2 is *"one integer with no parameters, training, or measurable compute overhead"* | fine-grained MoE | topk-2 of 1024, capacity 2.0 [ours b1.toml] | **ADOPT (trivial)** — one integer in the router | inference-side only; measure BPB on a frozen checkpoint before any training change |
| 13 | **Share First, Route What Remains** `2608.10392` 2026-08-11 | shared-expert extraction changes *what remains* and *how much capacity the remainder needs*; decompose FFN into KV channels and decide jointly | sparsely upcycled MoE | same block: shared expert + routed ranks | **READ-ONLY** — the decomposition is not landed here | reference for the b1 shared-expert sizing question |
| 14 | **Sparse probability maps** `2610.06677` 2026-10-05 | sparsemax/α-entmax/2-normmax promise token-dependent expert counts but "we study whether and how this sparsity survives training" | matched 300M and 1B top-2 MoE | directly our router shape | **READ-ONLY** — null-leaning; do not spend an A/B until the shared-expert probe lands | — |
| 15 | **Over-dispersed routing** `2609.04453` 2026-09-03 | under aggressive load balancing, router probabilities collapse as an importance signal; **perplexity does not predict downstream accuracy** on gpt-oss-20B | gpt-oss-20B | we run Load-Free Balancing at `moe_lb_coef = 0.01` [ours b1.toml] | **CAUTION** — add a router-entropy log; do not raise the coef | log `mean max-prob` per expert per step in the existing eval line |
| 16 | **Exact Quantile Balancing** `2609.28053` 2026-09-23 | exact global-batch BF16 quantiles + load-error injection into router gradients | 7.5B MoE | above our scale | **SKIP (scale)** — revisit at 1B only if drops are non-zero | — |
| 17 | **Raven** `2607.25357` 2026-07-28 | fixed set of memory slots, decay/update only a routed subset; extrapolates to **16×** training length, hybrid-friendly | recall benchmarks | an alternative to MSA's fixed recipe; not free | **SKIP (for now)** — MSA recipe is fixed by decision; keep as its fallback | — |
| 18 | **Attention sinks @1M** `2609.08574` 2026-09-08 | gated attention cut first-token attention 46.7% → 4.8%; Kimi K3 pairs it with **KDA + Attention Residuals** at a 1M window | diagnostic, 1M | our AttnRes exists but is off (b1 `use_attnres = false`) | **READ-ONLY (instrument)** — take the sink diagnostic into the depth-curve eval | — |
| 19 | **JSSA / PISA block-sparse** `2610.19026` 2026-10-13 · `2609.31093` 2026-09-25 | learned low-dim query–key subspace selection; pyramid top-k with log-linear selection cost | long-context decoding | alternative to the fixed MSA recipe | **SKIP** — re-open only if MSA fails its own gate | — |
| 20 | **UBE / UTF-16 boundary** `2610.22071` 2026-10-15 | writing 3-byte codepoints as UTF-16 low-byte-first gives the boundary predictor 8 informative low bits in the first symbol | multilingual boundary prediction | only matters once a patchifier exists | **SKIP-now, ADOPT-iff patching** — read with `2610.22100` | pair with #4 if a BLT arm ever lands |
| 21 | **Nested byte vocabs (negative)** `2608.28151` 2026-08-28 | prefix-nested byte-BPE vocabularies slice **numerically exactly** (76 checks), but sharing is expensive; pre-registered negative | 3.1M/10.6M bodies, 200M tokens, 30 models | byte-level ✓, cheap in principle | **SKIP (value)** — nice deploy trick, no quality lever | — |
| 22 | **Rust end-to-end LM report** `2609.25008` 2026-07-27 | five Candle defects (fused kernels with no gradient) and three Burn defects, incl. *"a backward pass at roughly 3% of theoretical GPU throughput"* and a fusion path that segfaults at multi-B parameters, at $164 total | one model, one GPU | our exact stack (Rust, burn, training a byte LM) | **READ-ONLY (required reading)** — independent confirmation of AGENTS §3.1/§3.2 | — |
| 23 | **Baseline-shape dominates** `2609.29397` 2026-09-24 | at 60K params, param-matched transformers span **22.6% validation loss purely by depth/width**; 98 byte-level runs, 3 seeds | 60K, M4-class | methodology, not a mechanism | **ADOPT (rule)** — cite when any single-seed arm is believed | already our 3-seed rule; this gives the published basis |
| 24 | **Best optimizer depends on batch size** `2610.08975` 2026-10-06 | *"no principled scaling rule for Muon works consistently across training settings"*; the best optimizer changes with batch size after tuning | LM pretraining | tiny batch, short horizon | **CAUTION** — do not import a Muon recipe from a large-batch paper | — |
| 25 | **Linear-attention survey** `2607.07953` 2026-07-08 | at 350M/15B: **KDA+Muon lowest final val loss**; pure GDN2+AdamW highest normalized throughput; hybrids improve loss at a throughput cost; Muon consistently lowers val loss vs AdamW | 350M/15B, plus 1.3B/3B DeltaNet | it is our arm, benchmarked against our family | **READ-ONLY (supports current arm)** — evidence *against* swapping KDA out | — |
| 26 | **Muon under real budgets** `2610.21555` 2026-10-15 | Muon cuts val-loss AUC 3.25–22.44% at equal data (17.7M/33.6M), but time-normalized AdamW wins on language by 3.38–6.96%; Muon call = 56.1–69.6% of step time | 17.7M/33.6M LM + 2.7M ViT, 1 A100, 5 seeds | our scale, but our step is launch-bound (opt 43–47/~245 ms) | **READ-ONLY** — keep the existing 2D-Muon/1D-AdamW split; do not extend Muon | — |
| 27 | **AMUSE (anytime Muon)** `2605.22432` 2026-05-21 | Schedule-Free and Muon combined: time-varying interpolation between the fast Muon iterate and the averaged iterate; no LR schedule | vision + LLM pretraining | pre-July; our WSD cooldown already exists | **SKIP (already answered)** — schedule-free was already evaluated in `2026-10-07-technology-super-research.md` (#3) | — |
| 28 | **LoopQuant / RAMP** `2610.16368` 2026-10-12 · `2610.13041` 2026-10-09 | loop-aware post-training quant (cross-loop activation dynamics); recurrent adaptive mixed precision for prefill | LoopLM deployment | we are not at low-bit deployment | **SKIP (stage)** — relevant only at deployment, like the quant ladder we already own | — |
| 29 | **Hybrid spec-decode: TreeWY / Bole / ReaDy** `2608.20961` · `2608.01651` · `2610.20503` (2026-08/10) | recurrent-state snapshot sharing, branch-wise verification, and drafting from HLA recurrent states | hybrid-attention decode | no generation-time spec-decode in our pipeline yet | **SKIP (stage)** — re-open with the DSpark row if byte spec-decode is revived | — |
| 30 | **Puro-2B / ufakzeka-1 / Daedalus-150M** `2608.27370`(covered) · `2609.25081` 2026-09-18 · `2608.20210` 2026-08-20 | single-GPU small-LM build reports ($509/$286, 151M Turkish, 150M conv-attn CPU model) | 150M–2B | budget reality matches ours | **READ-ONLY (comparison)** — honest baselines for our own cost claims | — |

## 2. TOP-3 deeper (STEP 3)

### 2.1 LOOM — `2610.01153` (2026-10-01) — the loop recipe we are missing
**What changes in our code/config.** Three named mechanisms, each a separate switch so they can be A/B'd
alone: (a) **bounded residual updates** — scale the per-visit residual by a bound derived from measured
hidden-state variance (we already have `residual_scale`, ReZero-initialised at 1; this makes it *dynamic* and
variance-aware); (b) **input re-injection at every loop** — we have `iter_embed (4×768)`, so the hook exists,
but `loop_block.rs:381` resets `h = h_ctx` each iteration, i.e. the block output never carries forward;
LOOM carries earlier outputs forward **and** re-injects `x`; (c) **per-loop routers** — our router is shared
across visits, which is precisely the "expert selection collapse" LOOM diagnoses. **[S1]** I expect (c) to be
the load-bearing one here, because our readout is already the mean of per-iteration outputs, so (b) is the
smaller perturbation.
**Risk.** Per-loop routers multiply router params by `max_iter` (trivial at 768→3, not trivial at 1024
experts) and add a gather per loop — our gathers are already the sm_120-hostile part of MoR/MoE
[ours, `2026-09-16-token-skipping.md`]. The 9–12-loop stability result is from another group's kernels. **Gate:**
if per-loop routers cost >10% step time at `small`, drop (c) and keep (a)+(b) — that arm still tests the
variance story.
**Cheapest gate.** `small`, 3 seeds, 2 000 steps, pure CE, one batch size across arms, tracking the depth
curve and φ (AB-PROTOCOL §verdict metric — BPB alone is void for a depth arm).
**Predicted outcome.** **[S1]** (a)+(b) reproduce at `small` (variance bound is topology-free); (c) is the
one that buys depth >2. **[S2]** at b1 the win is conditional on the MoE arm behaving — see 2.2.

### 2.2 MoE under repeated data — `2609.11917` (2026-09-10) — a gate, not a feature
**What changes.** Nothing in the model; it changes *what we are allowed to believe*. b1 is a 1.07B-total /
57M-active MoE [ours `configs/b1.toml`] trained on a 19.7 GB corpus that the recipe re-reads up to ~4×
[ours `2026-09-26-byte-lm-recipes.md`]. The paper's boundary — "MoEs begin to suffer at 4x" — sits exactly on
our corpus. If it transfers, every b1 MoE conclusion is measured in the one regime where MoE is documented to
lose its advantage, and the "MoE null result" in `2026-10-16-token-skipping.md` needs a repetition-controlled
re-read.
**Risk.** The paper's own remedy (*"strong masking-based regularization"*) is an extra training knob that
confounds a clean A/B; and its scales (80M–1B active) are above ours, so the 4× threshold may not bind at
57M active. **[S2]**
**Cheapest gate.** Take the existing data-repeat ladder and run it **with the MoE arm ON** for one seed per
rung (unique / 2× / 4×), reading BPB on a fixed eval window. Cost is existing infrastructure, no new code.
**Predicted outcome.** **[S1]** the 4× rung is where a gap appears; **[S3]** we cannot say whether masking
regularization closes it at our active-width.

### 2.3 Complex KDA — `2609.24797` (2026-09-21) — the cheapest real adoption
**What changes.** Two parameter ranges on the KDA arm we already run: allow the channel gate into `[-1,1]`
(a reflection, not only a decay) and the delta coefficient `β` into `[0,2]`. The paper proves the resulting
recursive matrix is still diagonal-plus-rank-one and non-expansive, so the stability argument that keeps KDA
in the model is untouched. Our port's KDA already carries `q,k,v,g` and a separate `beta` [ours small.toml
param table], so this is a clamp + a range, not a new arm.
**Risk.** FLA's chunked KDA kernel assumes a log-decay gate; a signed gate is a **different kernel**. On the
`port` side it is ours to change; on the burn/FLA side this is where the cost actually lands. **Gate:**
assert non-expansiveness numerically (spectral bound) before any BPB row — a violation is a silent NaN source
of the class already documented in this repo.
**Cheapest gate.** `small`, 3 seeds, 2 000 steps, pure CE, KDA arm otherwise byte-identical; read BPB at two
depths so length extrapolation is visible.
**Predicted outcome.** **[S1]** small BPB gain at `small`, larger at the 100M rung; **[S2]** the real payoff
is length extrapolation (the paper's strongest evidence is `S₃/S₄` + periodic continuation), which our 2 048
context cannot show.

## 3. Rejected with reason (so nobody re-screens them)
- `MuonClip` · `LoMu` · `PolarGrad` · `delta-product` — **no real id found** (dedicated `abs:` queries, 2026-01-01→2026-10-16, zero hits). MuonClip exists only inside the Kimi K2/K3 tech reports, already covered.
- `schedule-free Muon` — the only real hit is **pre-July**: AMUSE `2605.22432` (2026-05-21); schedule-free itself was already evaluated (`2026-10-07-technology-super-research.md` #3).
- `PSGD` successors / "second-order-lite" — no 2026-07+ successor surfaced; `2610.04204` (Clean, Nyström sketched second order) is the nearest and is memory-shaped, not LR-shaped.
- Byte-level spec-decode newer than `2610.05978` — **none found**; `2610.20503` ReaDy is hybrid-linear-attention drafting, `2610.20865` is long-context drafting for normal transformers.
- `2609.38166` LeapQuant / `2609.38169` STEPQuant / `2608.27513` DAMP / `2610.15709` — recurrent-**state** quantization; the quant ladder is already swept, and these are inference-stage.
- `2608.06628` (retrofit linear attention into dLLMs), `2609.34253` (AR→gated-delta diffusion distillation) — different output family, retrieval does not transfer (the paper says so).
- `2609.02404` shared routing geometry · `2608.21840` more experts worse dynamics · `2609.03501` statistical MoE — theory/caution, no lever at our scale.
- `2609.18176` MoRE · `2609.22199` CS-MoE · `2610.14644` LatentMoE shapes — cross-layer expert sharing; plausible but strictly behind the shared-expert exploration probe (#11).
- `2610.05700` Voltic · `2609.36259` CyFA · `2609.18407` TERN · `2609.35200` ReCAT — delta-rule/state variants in non-LM domains (audio, epidemics, robotics); no LM evidence.
- `2607.19390` orthogonalized read scaffold · `2607.17419` KATA · `2609.02986` implicit hybrids · `2609.33051` SketchSSM · `2608.30386` DASC · `2609.14320` SpectralShift · `2610.16268` TTT error correction — read, real, but each is either a serving-side trick or a mechanistic study with no measured LM win at ≤1B.
- `2608.19197` SPADE self-play · `2609.30535` code-switch curricula · `2608.27370` Puro-2B — self-play is RL-stage (our pretrain is corpus-bound), and Puro-2B is already covered.
- `2610.12448` recurrent ViT depth-programmed experts — vision, but the closest non-LM analogue of #7; noted, not screened further.

## 4. Open questions
1. Does the MoE repetition boundary (@4×) bind at **57M active**, or is it a large-active-param phenomenon? Only our own ladder answers it.
2. Is our φ = 0.46 recovered as a special case by `2609.40316`'s joint law once sparsity is a factor? If not, one of the two fits is wrong at our scale.
3. Does our KDA arm have latent **expert-selection collapse**? We have never logged per-loop router overlap; #11 is blocked on that measurement, not on an A/B.
4. Does the port's KDA allow a signed gate without a new kernel? If not, Complex KDA costs a kernel, not two lines.
5. We have not verified the LOOM per-loop-router cost on sm_120 — the one number that decides whether #1 ships as (a)+(b) or (a)+(b)+(c).
