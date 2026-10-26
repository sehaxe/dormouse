# Adaptivity in the loop we already run — what is published, what it costs on this card, what to A/B

**Date:** 2026-10-21. **Read-only:** this file only; no config or code touched, nothing committed
(the `docs-site/tools/manifest.mjs` line is owed, last line of Sources).
**Question (owner):** the architecture is "full, adaptivity included", but every arm must win a
measured A/B or be deleted. A previous internal verdict said MoD/mixture-of-depths is NO-GO at our
scale (FLOP skipping saves ≤4-7 % because the loop is launch-bound). Re-examine that as a
hypothesis. Specifically: resolve "looped world models", survey adaptive compute as published in
2026, price each mechanism on OUR card, rank by fit, pre-register a ladder, and name the rejections.

**Method.** 13 `search_papers` queries + one `get_abstract` per cited ID + 3 `download_paper`
(2606.18208, 2502.05171, 2507.10524) with `read_paper_section` / `search_paper_text` on each. **No
HTTP 429 occurred; no export-API fallback was needed.** Repo facts were read from the code and from
`docs/reviews/launch-atlas-2026-10-02.tsv`, not assumed.

**Marks:** **[T]** full text read this session · **[A]** abstract read in full · **[S]** snippet only
(title + ~240 chars) · **[ours]** our file:line or our measurement · **[S1]** SPECULATION with a
sourced mechanism, transfer unmeasured · **[S2]** SPECULATION with no source.

## 0. What the owner asked, and what the words resolve to

1. **"Looped world models" is a real paper, not a paraphrase.** `ti:"looped world model"` returns
   exactly one hit: **2606.18208, "Looped World Models" (LoopWM), 2026-06-16, FaceMind Research
   Asia** [T]: a parameter-shared transformer block iteratively refines a latent environment state,
   plus a per-iteration sigmoid exit gate. **It does claim that loops improve world-model
   behaviour** — ScienceWorld next-state prediction at ~1B params: **EM 68.4 / F1 85.3 / BLEU-4 80.7**
   against **claude-opus-4-6-max at 47.2 / 72.8 / 64.4** (+21.2 EM) [T §6.1, Tables 2-3]. The caveat
   that matters: **there is no depth ablation in the paper** (`search_paper_text "ablation"` → 0
   passages), so that +21.2 EM is loops + params + objective + data, undifferentiated. "Loops improve
   world-model behaviour" is the authors' framing, not an isolated measurement. Closest relatives:
   Huginn (2502.05171, random log-normal-Poisson depth, 3.5B/800B tokens) [T §4.3], Universal
   Transformers (1807.03819, weight sharing + ACT) [A], DEQ (1909.01377) [A]. The Dreamer line is
   *recurrent in time*, not looped in depth.
2. **"Adaptivity" resolves to three mechanisms the literature conflates.** They are not
   interchangeable, and the old NO-GO is only about the second:
   - **(i) iteration-level skipping** — one scalar decides whether iteration *i* runs for the whole
     batch. This is what our MSA arm does; it removes that iteration's launches *and* its FLOPs.
   - **(ii) token-level skipping** — per-token depth (MoD, MoR, T-LoopFormer). Removes FLOPs, keeps
     or adds launches.
   - **(iii) test-time depth scaling** — run *more* iterations than trained at inference: spends
     compute, saves nothing, and the only one that can *raise* quality.
3. **Our own numbers separate (i) from (ii), and that is the answer to the re-examination.** At
   `small`/b8/s512 a warm step is **21 427 launches / 581.5 ms**, and `--max-iter 4→2` removes
   **7 778 launches (36.3 %)** and **284.9 ms** — one loop iteration is **18.1 % of the launches and
   ~24.5 % of the wall-clock** (`launch-atlas-2026-10-02.tsv` rows `base`/`iter2`; delta table at
   `launch-atlas-2026-10-02.md:100`), while `opt`/`retr` have **no per-iteration component** (`:116`).
   So a skipped iteration is worth ~142 ms at small, not 4-7 %: that figure belongs to (ii), where
   the launches stay. **The prior verdict is right about MoD, wrong only if generalised to
   iteration-level depth.** But §1.3 forbids a host read in the step, and a decision the host never
   reads cannot change the next launch — so at **train time** "adaptivity" collapses to a
   **schedule** (a smaller `max_iter`, or a gate that multiplies the readout), and the *dynamic*
   version of (i) exists only at **inference**. That is the re-examination in one sentence.

## 1. Verified paper table (every ID resolved by `get_abstract` this session)

| id | title | date | claim that matters here | measured at | compute-matched control? |
|---|---|---|---|---|---|
| **2606.18208** [T] | Looped World Models | 2026-06-16 | looped block + per-iteration sigmoid exit gate; "up to 100× parameter efficiency"; **25× FLOP reduction per simple step, "up to two orders of magnitude" aggregate** (§4.4) | ~1B params, ScienceWorld next-state (5 actions), vs API models | **no**; the 25×/100× are projections in §4.4, no τ sweep, no depth ablation |
| **2502.05171** [T] | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn) | 2025-02-07 | iterate a shared block; random iteration count per sequence (**log-normal Poisson**); truncated BPTT through the last `k=8` | 3.5B params, 800B tokens; "close to what a 32B transformer would consume" in pretraining FLOPs; improves up to **50B-equivalent** test-time FLOPs | no; it is compute-*spending*, not saving. Own caveat: 47k steps, LR never cooled |
| **1807.03819** [A] | Universal Transformers | 2018-07-10 | weight sharing across depth + ACT halting | MT/algorithmic, small | no |
| **1909.01377** [A] | Deep Equilibrium Models | 2019-09-03 | infinite-depth weight-tied net via root-finding; constant memory | WikiText-103, small | no |
| **2404.02258** [A] | Mixture-of-Depths | 2024-04-02 | top-k token routing through a block, static graph, known tensor sizes; **parity at isoFLOP**; +~1.5 % on the training objective and 60M-3B scale as recorded in `2026-10-16-token-skipping.md`, which read it in full | 60M-3B | **yes** (isoFLOP) |
| **2107.05407** [A] | PonderNet | 2021-07-12 | geometric halting distribution; loss = expected per-step loss + KL | parity/sort, bAbI QA, enwik8-scale | partially ("matched SOTA using less compute") |
| **2207.07061** [A] | Confident Adaptive Language Modeling (CALM) | 2022-07-14 | per-token early exit with a sequence-level constraint; **"potential speedup of up to ×3"** | T5 220M-770M, 3 generation tasks | no — "potential", not realised latency |
| **1603.08983** [A] | Adaptive Computation Time | 2016-03-29 | learned per-step halting, differentiable | parity/sort/adding; **char-LM: "does not yield large gains"** | no |
| **2507.10524** [T] | Mixture-of-Recursions (MoR) | 2025-07-14 | per-token recursion depth + recursion-wise KV cache; **Pareto at equal training FLOPs** | **135M/360M/730M/1.7B, 3 recursions (≈1/3 unique params)**, budgets 2e18/5e18/16.5e18 | **yes** (equal-FLOPs vs vanilla and recursive) |
| **2607.14427** [A] | Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers | 2026-07-15 | per-token fixed point; **training-free stability halt = depth-8 quality at 4.94 avg loops (−38 %)**; a *learned* linear router on convergence labels yields **no** reduction | **135M-class, FineWeb-Edu, one RTX 4090, ~100 GPU-h** | **explicitly no** — "average depth as a FLOP proxy … no FLOP-matched parity claim"; single scale, single seed |
| **2607.20519** [A] | Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates | 2026-07-08 | joint gate training corrupts the *trajectory*; **post-hoc confidence readouts match or beat learned linear/MLP gates**; fixed-prior depth supervision produces the useful trajectories | synthetic + Ouro-1.4B/2.6B | no, but the comparison is gate-vs-readout on the same checkpoint |
| **2603.23701** [A] | The Diminishing Returns of Early-Exit Decoding in Modern LLMs | 2026-03-24 | early-exit benefit **falls across model generations**; **dense > MoE and > SSM** in exit potential (the paper does not order MoE against SSM); >20B shows more | modern LLMs (>20B emphasised) | n/a (negative result) |
| **2609.34212** [A] | X-MoD: Practical Scaling Laws for Sparse-Depth Routing | 2026-09-28 | sparse-depth law fitted **relative to FLOP-matched dense baselines**; decouples sparsity from anchor stride | pretraining sweeps, held-out law | **yes** (FLOP-matched dense) |
| **2610.23036** [A] | Towards Looped Models Done Right, Part I | 2026-10-16 | Huginn-style beats Ouro **on all 10 benchmarks at 730M/336B**; under matched train+inference FLOPs, **75 % fewer resident params** than a 112-layer FFN MoE | 730M/336B; 8BA0.8B/500B | **yes** (matched params, depth, tokens, FLOPs) |
| **2610.25086** [A] | Dual-Slot: From Serial Loops to Parallel Slots | 2026-10-18 | removes the serial latent dependency across decode steps; 1B dense **Pile PPL 7.25 → 7.00**, **+2.26 pp** avg 5-shot; keeps **93.1-96.5 %** of decode throughput | 1B dense / A0.6B MoE | no, but throughput is measured |

Snippet-only, not load-bearing: 2609.34488 FlexLoop · 2609.37432 Looped Actor · 2609.15160 T-LoopFormer ·
2602.11451 LoopFormer · 2610.25646 SFPRM · 2606.16112 NAG · 2607.06601 TriRoute [S]. Also read, not rowed:
**2610.28540** [A] — a small next-latent loss does not certify free latent rollout (open-loop error grows
linearly); a caution against reading "looped latent = world model" too literally.

## 2. The honest cost model on OUR card

**Two regimes, both measured, and the repo names the boundary** ("overhead-bound below ~80M params,
GEMM-bound above", AGENTS §2.2):

| regime | evidence | per-launch cost | what a saved FLOP is worth |
|---|---|---|---|
| launch-bound | `small` 9.2M, b8 s512: 21 427 launches / 581.5 ms ≈ **27 µs of step per launch**; GPU 13.3 % utilised, 142/180 samples ≤5 % (AGENTS, 2026-09-29) | ~27 µs | **nothing**, unless the launch disappears with it |
| compute-bound | `medium` 138M, b16 s1024: fwd 194.5 / bwd 282.0 / clip 18.3 / opt 78.3 = **573 ms, GPU 100 %** (`2026-10-18-throughput-leftovers.md:46`, passed in as "given", not independently measured) | hidden | ~1:1 |

**The one number:** one loop iteration ≈ **142 ms at small** and ≈ **119 ms at medium**
((194.5+282.0)/4) — a skipped *iteration* is worth 20-25 % of the step, a skipped *token row* is a
fraction of one kernel (§0.4).

**But the sync cannot be priced at 20 µs: AGENTS §1.3 forbids it, and the graph lane depends on it.**
A per-step scalar read serialises host against GPU *and* makes the step uncapturable
(`cubecl-runtime/src/client.rs:1288-1296`), and capture is the one lever against the 87 % idle. The
split is architectural: **at train time the host must not read the depth; at inference it may.** At
train the gate can only *multiply* the readout (CE gradient, no branch) or the depth is a
**precomputed schedule** — a constant, like `rand_depth`'s pure function of `(seed, step)`; a
device-side mask saves the arithmetic but not the launches. At inference (decode, and the eval loop)
host-side halting is free relative to 119-142 ms per iteration, but per **token** it is not:
4 096 × 10 µs = 41 ms = 29 % of an iteration.


| mechanism | extra kernels / step | host reads (train / infer) | FLOPs | wins in which regime | new state? |
|---|---|---|---|---|---|
| **iteration skip, static schedule** (our MSA gate) | **−** one iteration's share (~3.9k launches at small) | 0 / 1 per iteration | −18-25 % | **both** | no |
| **training-free stability halt** (‖Δh‖ or output KL per iteration) | +1 reduction +1 compare per iteration | **not allowed at train** (§1.3) / 1 per iteration | −(skipped iterations) | inference only | no |
| **learned per-iteration gate** (LoopWM shape: `σ(wᵀh+b)`) | +2-3 tiny ops per iteration | 0 at train (multiplies the readout) / 1 per iteration | −(skipped at infer) | both, if used as a schedule at train | `w∈R^d` |
| **per-token depth** (MoR/MoD/T-LoopFormer) | +top-k, +gather, +scatter **per iteration**; dropless MoE keeps expert GEMMs dense, so the saving is only the skipped rows' attention/FFN | 0 (mask) at train / 1 per iteration | −(skipped rows) | **compute-bound only**; at small it is negative | router params + per-token depth bookkeeping |
| **PonderNet** | +1 halting head +1 loss term | 0 at train / 1 per token at infer | −(halted steps) | both | `p_dist` machinery — deleted twice here (ADR-0013) |
| **early exit at decode** | +1 head | — / 1 **per token** | −(layers not run) | decode only | exit head |
| **test-time depth scaling** | 0 | 0 | **+** | neither — it spends | no |

**Why MoR's published throughput does not transfer.** MoR attributes its throughput to
*"recursion-wise key–value caching … reduces memory traffic"* plus grouping different-depth tokens
into one batch to remove bubbles (`search_paper_text 2507.10524`, "throughput") [T]. **Our KDA arm
has a constant-size state per head and no growing KV cache**, so the memory-traffic half does not
exist here, and the bubble half needs exactly the per-token machinery that adds launches. Expect
MoR's 135M Pareto point to be flatter for us — **our inference, not a published result** [S1].

## 3. Ranked fit for our loop

Ranked by *(published effect at ≤1B × fit to our loop × cheapness)*. A/B cost = `medium` b8·s768,
2 000 steps ≈ 12.3 MB ≈ 5 min/seed (`2026-10-18-data-capacity.md:26`, 39.5 kB/s → 7.0 GPU-h/GB).

| # | mechanism | source (mark) | published effect | fit to our loop | new kernel? | A/B cost | verdict |
|---|---|---|---|---|---|---|---|
| **1** | **Training-free per-iteration stability halt at inference** | 2607.14427 [A] + 2607.20519 [A] | −38 % average depth at depth-8 quality, 135M, single 4090 | **perfect** — `rand_depth` already trains every depth 1..`max_iter`, the readout is already a mean over executed iterations, and `decode` already recurs | **no** | **0** (reuses the control's checkpoints) | **build the measurement first; inference only (§1.3)** |
| **2** | **Read the depth curve we already can read, then add depth to decode** | 2502.05171 [T] | quality keeps rising with depth up to 8-32 at 3.5B; our trained range is 1..4 | **perfect** — `--eval-depths` already prints BPB at 1..=`max_iter` for free (AGENTS §3.7); what does **not** exist is depth on the *decode* path (`train.rs:131`: "there is no flag to lower it") or any depth > trained | no | 0 (eval-only) | **free measurement now; `--gen-depth` later** |
| **3** | **Per-iteration gate on the core block** (the MSA gate's shape, extended to all iterations) | 2606.18208 [T]; our MSA 27.6 % (owner-reported) | 27.6 % of iterations skipped at seq 2048 with a quality tie **on our own model** | high — one scalar per iteration; at train it can only *multiply* the readout (no branch, no readback), the skip being an inference decision | no | ~15 min × 3 seeds | **extend, but expect a schedule not a router** |
| **4** | **Per-token depth / MoR** | 2507.10524 [T] | Pareto at 135M-1.7B **at equal FLOPs**, mostly via KV traffic | medium — `mor.rs` exists and has never had an A/B (smoke only: `mor_ab.sh preflight` 1068 ms/step) | no | ~3.5 GPU-h for 3×2 | **run it as the falsification of the prior NO-GO** |
| **5** | **More-depth-at-fixed-wall-clock control** (8 iterations, half the steps) | 2610.23036 [A] | matched-FLOPs looped ≈ 112-layer FFN with 75 % fewer resident params, 730M | high — answers "where does the second unit of compute go" | no | ~30 min × 3 seeds | **required control, not an arm** |
| **6** | **PonderNet / learned halting with KL(p_dist)** | 2107.05407 [A]; ADR-0013 [ours] | matched SOTA "using less compute", small scale | low — deleted here after **two measured collapses**; 2607.20519 explains the mechanism (joint gate training degrades the trajectory) | no | n/a | **reject (§5)** |
| **7** | **MoD token-level skip** | 2404.02258 [A] | parity at isoFLOP; "+50 % faster to step" at 12.5 % capacity | low in the launch-bound regime | **yes** (sparse-gather kernels) | n/a | **reject (§5)** |
| **8** | **Early exit at decode** | 2207.07061 [A]; 2603.23701 [A] | ×3 *potential*; but MoE+SSM have the **least** exit headroom | low — our decode is a serial recurrence, and our class is the one the negative result names | no | n/a | **reject for now (§5)** |
| **9** | **Dual-Slot latent/prediction split** | 2610.25086 [A] | +2.26 pp at 1B, 93-97 % throughput retained | medium — not adaptivity, but it buys "more loops without more latency" | no | ~1 day | **keep as the architectural alternative if (2) shows depth saturation** |

## 4. Pre-registered ladder — three arms, three seeds, 2 000 steps

Fixed: `medium` (138 435 969 stored params, d1024, E=256 top-2, rank 64), b8·s768, lr schedule
unchanged, same corpus shard, `eval_batches` fixed and the byte count printed. Bar = **the control's
own 3-seed range** (`AB-PROTOCOL`); expected effects are **OUR hypotheses**, not the papers'.

- **A — control.** fixed depth 4, no gate. 3 seeds × 2 000 steps ≈ 15 min.
- **B — stability halt at inference.** *Same weights as A, bit-identical training step* — the halt
  runs in the **eval loop and `decode`**, never in the step (§1.3; the read would also make the step
  uncapturable). Rule: after iteration `i`, compute `‖h_i − h_{i−1}‖ / ‖h_{i−1}‖` and stop below τ;
  τ swept on a held-out slice, reported as a curve, never tuned on eval. **Expected:** 20-35 % of
  iterations skipped at ≤ +0.005 BPB (anchor: 4.94/8 = −38 % at 135M, 2607.14427; our ceiling is
  3/4). **Kill:** best-τ BPB worse than A by more than A's seed range, **or** skip < 10 % at parity —
  the latter means the trajectory is not difficulty-aware and (1) is dead here.
- **C — more-depth-at-fixed-wall-clock control (compute-matched).** `max_iter 8`, **1 000 steps**
  (same wall-clock as A at ≈1.8× step cost — `opt`/`retr` do not scale with iterations), same d1024.
  The confound is inherent and is the question: at fixed wall-clock, 2× compute per step forces half
  the steps and hence half the tokens — *where should the second unit of compute go*, not "does
  depth help at all". **Expected:** if loops are a real quality axis, C beats A by 0.01-0.03 BPB; a
  tie means loops past 4 are saturated at our data budget. **Kill:** C ≤ A + seed range ⇒ depth
  beyond 4 earns nothing, only the latency case survives, and (1)/(3) are the only rows left.
- **D — per-iteration gate.** depth 4 + `σ(wᵀh_i+b)` per iteration. At train the gate only
  **multiplies** that iteration's readout (a scalar weight on a tensor — no branch, no readback, so
  §1.3 holds and `w` gets CE gradient with no sampling term); the **skip** is inference-only.
  `--rand-depth` and `use_mor` are mutually exclusive by a loud refusal in `resolve` (AGENTS §3.3),
  so D runs `rand_depth` on, MoR off. **Expected:** ≥25 % skipped at tie (anchor: our MSA 27.6 %).
  **Kill:** skip < 15 %, or BPB worse than A beyond the seed range — then the gate is a schedule, not
  a router, and the MSA arm already is one.

**Resolution warning, stated before the run.** At 2 000 steps a `medium` run has seen ~12.3 MB. The
project's own estimate of **paired-eval resolution is 0.002-0.005 BPB and is explicitly unverified**
(AGENTS §3.1); the `small` preset's 3-seed spread is 0.048-0.073 BPB and no `medium` spread has been
measured. So the effects above sit **at** the resolution floor: this is a **screening** ladder, a TIE
is a deletion, and only a survivor earns the confirming 10 000-step run (61 MB, ~50 min/seed).

**The measurement that outranks all four arms (cost: one eval).** `--eval-depths` already prints BPB
at depths 1..=`max_iter` for free (AGENTS §3.7) — **read that first, on any `medium` checkpoint, before
writing a line of code.** Then the part that does not exist: a **free oracle**, per-token best depth's
CE summed over the eval window. If `oracle − best_fixed_depth` is inside the seed range, adaptive
depth at 138M cannot pay for itself no matter which gate wins: the cheapest kill here, and it runs
before D.

## 5. Explicit rejections

| rejected | why (sourced) |
|---|---|
| **PonderNet / learned halting with a halting distribution** | Deleted twice here for measured collapses (ADR-0013); `p_dist`/KL machinery is gone. 2607.20519 [A] supplies the mechanism: joint gate training changes the *trajectory* it is trying to read, and post-hoc readouts match or beat learned gates. Re-entry only as arm D (a readout gate), never as PonderNet. |
| **MoD (token-level, 2404.02258)** | The one regime where our prior verdict is correct: token skipping keeps every launch, and at `small` a launch costs ~27 µs of step. The paper's own speed number needs 12.5 % capacity every-other-block and sparse kernels we do not have. Parity, not a win, in its own isoFLOP table. A *host-side* per-token decision is worse: 4 096 tokens × 10 µs = 41 ms = 29 % of the 142 ms an iteration is worth, so saving and decision cancel. |
| **Early exit at decode** | 2603.23701 [A]: exit benefit **falls** across model generations and is **lowest for MoE and SSM-like** models — exactly our class. CALM's ×3 is "potential" on T5-220M/770M. Our decode is serial per token, so a per-token sync costs real time at byte granularity. |
| **A fused variable-depth loop kernel** | Blocked by a live, unfixed defect: cubecl's launch path silently **drops tail cubes for certain output shapes** (`d2_isolate`, AGENTS 2026-09-30), and our fused kernels already refuse shapes below `MIN_FUSED_D=4` as a *measured* workaround, not a law. A kernel whose output shape changes with the depth decision is the worst possible input to that bug. At train time depth must be a **constant schedule**, at inference a host decision; kernels stay fixed-shape either way. |
| **Per-step dynamic skip during training (a host read in the step)** | Forbidden by AGENTS §1.3 (zero host-device sync in the hot path) and it would make the step uncapturable (`client.rs:1288-1296`), killing the CUDA-graph lane that is the only attack on the 87 % idle. A device-side mask can skip the arithmetic but not the launches, so it buys ~nothing in the launch-bound regime. **This is the structural reason adaptivity here can be a schedule at train time and a router only at inference.** |
| **Stochastic per-token halting (sampled Bernoulli / MCMC-ish)** | Breaks the property that a step is a pure function of (seed, step) — the property `rand_depth` was built to have (ADR-0021), and the one that makes resume comparable. Adds a sampling term whose variance we cannot afford at 2 000 steps. |
| **LoopWM's headline multipliers (25×, 100×, "two orders")** | Not measurements: §4.4 gives an illustrative 100-layer baseline and multiplies, with no τ sweep, no depth ablation, no latency table [T]. Citing them as expected effects would be fabrication by proxy. |
| **X-MoD / TriRoute / NAG as arms** | 2026 sparse-depth *architecture* papers whose wins are relative to FLOP-matched dense baselines at scales and sparsity patterns we cannot reach; X-MoD's law needs "anchor stride" layers we do not have. Read as evidence, not as arms [A]/[S]. Out of scope besides: multi-GPU or large-batch adaptive serving — every serving-side argument (bubble elimination, cross-request batching of different depths) needs a batch this card does not have. |

## 6. What is NOT verified

1. **No published measurement of adaptive-depth wall-clock at ≤1B on a single consumer GPU** other
   than 2607.14427, which **says so itself**: average depth is a FLOP proxy, wall-clock a three-point
   bracket, one scale, one seed, "no FLOP-matched parity claim". The rest is ≥1.7B (MoR), 3.5B
   (Huginn), ~1B with no ablation (LoopWM), or an API-model comparison.
2. **Our 27.6 % skip at seq 2048 is owner-reported and I could not reproduce it from this
   checkout.** `grep -rn msa_gate crates/` returns nothing; ADR-0014 cut the MSA arm and `use_msa`
   survives only as a distillation stage (`config/schema.rs:616`); AGENTS §3.3 records that
   `--gen-max-iter` **does not exist** and that ADR-0013's CALM-lite confidence exit was **never
   implemented**. So the gate the brief describes is not in this tree. It is the one local
   measurement of iteration-level skipping and the most load-bearing number in §3 row 3; **its log
   path and branch should be attached before the ladder runs** — and §1.3 implies it can only have
   been an inference/eval-side measurement.
3. **Both of our own base numbers are softer than they look.** The 573 ms `medium` profile is
   "given", not measured by the doc that prints it (`2026-10-18-throughput-leftovers.md:46`), and the
   4→2 iteration delta is an **atlas comparison**, not a pre-registered A/B: `base` and `iter2` are
   separate runs and their flags are not proven identical, even though `opt`/`retr` are
   iteration-independent by construction (`:116`). The `small` numbers (21 427 launches, 581.5 ms,
   13.3 % util) are measured, on this box; the per-iteration marginal cost is the best *local*
   evidence, not a controlled experiment.
4. **Ouro has no arXiv ID I could resolve.** `ti:Ouro` returns 0; the 1.4B/2.6B checkpoints are
   cited by 2607.20519 and compared against by 2610.23036, so every Ouro claim here is second-hand.
5. **Whether 2607.14427's −38 % survives byte-level tokenisation** is unknown: byte sequences are
   3-4× longer than BPE token sequences, so a per-token halt rule fires on a different unit and
   "median token converges by loop 6" has no byte-level counterpart. Largest transfer risk here.
6. **No source measures the interaction we actually have**: weight-shared loop + KDA linear
   attention + dropless top-2 MoE + Engram, at 138M, on one 16 GB card. Every §3 fit is a transfer.

## 7. Open questions

1. Does the `medium` 3-seed spread at 2 000 steps exceed 0.01 BPB, and is `oracle − best_fixed_depth`
   above that spread? Nothing measured either; both cost one short run, and the whole ladder's power
   depends on them.
2. Does depth 5-8 (never trained) degrade or hold, and does the MSA gate's 27.6 % survive being
   moved from the sparse-attention arm (where a skipped iteration saves little, since KDA still runs)
   to the whole block (where it saves 142 ms)? `rand_depth` trains 1..4 only, and Huginn sampled
   *including* the target range — so extrapolation may simply fail, which is why arm C trains
   `max_iter=8` rather than testing it post hoc.

## Sources (arXiv IDs; how each was verified)

**Full text [T]** — `download_paper` + `read_paper_section` / `search_paper_text`: **2606.18208**
(§4.4 gate, §6.1 tables, "ablation"→0 hits) · **2502.05171** (§3.3/§4.3 log-normal Poisson, k=8
truncation, §6.1 32B/50B-equivalent) · **2507.10524** (v3; "throughput" and "135M" passages).

**Abstract [A]** — `get_abstract`, full abstract read: 1807.03819 · 1909.01377 · 2404.02258 ·
2107.05407 · 2207.07061 · 1603.08983 · 2607.14427 · 2607.20519 · 2603.23701 · 2609.34212 ·
2610.23036 · 2610.25086 · 2610.28540 · 2606.16112 · 2610.25646 · 2602.11451 · 2609.15160.

**Snippet [S]** — title + ~240 chars: 2609.34488 · 2609.37432 · 2609.15160 · 2602.11451 ·
2607.06601 · 2610.08830.

**Wrong-ID log.** `2407.00611` was my guess for Mixture-of-Recursions; `get_abstract` returned
**StarTrail** (concentric-ring sequence parallelism). The right ID was found by
`ti:"Mixture of Recursions"` → **2507.10524**. Similarly `ti:Ouro` → 0 hits (see §6.4), and
`ti:"elastic depth"` → 3 hits, only one of them a looped LM (2602.11451).

**Repo [ours]:** `AGENTS.md` §1.3 (no host sync in the step), §2.2 (overhead-bound < 80M),
§3.1 (paired-eval resolution 0.002-0.005 BPB, unverified; capture refuses stream reads), §3.3
(`--gen-max-iter` absent; ADR-0013's CALM-lite never implemented; `rand_depth` ⊥ MoR), §3.7
(`--eval-depths`, `--rand-depth`) · `train.rs:127-133` ("there is no flag to lower it") ·
`schema.rs:135` (`max_iter=4`) · `schema.rs:616` (`use_msa`) ·
`docs/reviews/launch-atlas-2026-10-02.{tsv,md:100,116}` · `2026-10-18-data-capacity.md:26,40` ·
`2026-10-18-throughput-leftovers.md:46` (the "given" 573 ms) ·
`2026-09-27-adaptive-depth-safe.md` (the same depth-curve-first conclusion) · ADR-0013 · ADR-0014 ·
ADR-0021. `docs-site/tools/manifest.mjs` is **not** updated for this document (task restricted to one
file); `tools/check_doc_refs.py` will flag it until that line lands.
