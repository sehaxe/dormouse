# Expert collapse and load balancing at 199M params / 512 experts / byte level

**Date:** 2026-11-04
**Scope:** literature review + arithmetic on the reported run. No code touched.
**Config under review:** byte-level looped MoE, 199M params, d=1536, E=512, top-2, rank 64, shared input basis `share_u`, dropless, 150k steps, FineWeb-Edu.
**Reported:** 80/512 experts receive mass on real bytes (15.6%); top-1 = 8.3%; top-10 = 50.1%; random bytes = 439/512. LFB (arXiv:2408.15664) with sign update, coef 0.01, counters reset across 5+ resumes.

---

## TL;DR

1. **The measurement is the first suspect, not the model.** LFB only moves the *selection* bias; it is explicitly not applied to the gate weights (arXiv:2408.15664, 2024-08-28, §4). If the histogram is computed from gate mass rather than hard top-k counts, LFB can be working perfectly and still show 80/512 by mass. This one distinction changes the diagnosis completely. It is a one-line check and must precede any model change.
2. **At 8k tokens/batch and top-2 you have 32 assignments per expert, not 16** (8192×2/512 = 32.0). At that count the per-batch estimate of "is this expert overloaded" has ~18% Poisson noise, so a sign-integrator with γ=0.01 is mostly integrating noise. The reference implementations use γ=0.001 (DeepSeek-V3, arXiv:2412.19437, 2024-12-27, §5.3) and LFB's own tuned 1B value is 0.001 (arXiv:2408.15664, Table 3).
3. **15.6% alive is not a documented steady state, but the literature almost never reports it.** The one paper that plots live experts over training at a comparable scale (230M active, 64 experts) reports *all* experts used across the validation set, with per-sequence utilization 0.38–0.51 under LFB and 0.997 under an aux loss (SimBal, arXiv:2506.14038, 2025-06-16, Table 8). No paper reports a live-expert fraction for E=512 at 200M params. That absence is itself a finding.
4. **At byte level there is a benign alternative explanation for 80/512.** OpenMoE found routing is predominantly determined by token ID, not context (arXiv:2402.01739, 2024-02-01, §1, §5.2). At byte level the "token ID" is the byte, so the alphabet is 256 and English byte frequencies are steeply skewed. One expert owning ~8% of tokens is exactly what "this expert owns the letter `e`" looks like. This is testable in one forward pass (mutual information between byte identity and chosen expert set) and is the highest-value experiment in this memo.
5. **Shrinking the bank is a real option, but "strictly better" is false.** With total expert params = E × size and active FLOPs = k × size, the active ratio is k/E. Holding total params *and* active FLOPs fixed, the active ratio cannot be raised. E=128 at top-2 keeps active FLOPs identical and gives 4× more tokens per expert, at 4× less total capacity. Whether that wins depends on whether capacity is binding — and at ~6 bytes/param it probably is not.

**Confidence in this verdict: medium.** The mechanism literature is solid. The specific diagnosis of *this* checkpoint is inference until the hard-count-vs-mass check is run.

---

## 0. Measurement forensics (do this before anything else)

Everything below is arithmetic on the numbers in the brief, not a literature claim.

**0.1 Tokens per expert.** 8192 tokens × top-2 = 16 384 assignments. Divided by 512 experts = **32.0 per expert**. The brief says 16, which is 8192/512 — the top-2 factor is missing. This matters because every noise argument scales with it.

**0.2 Is 80/512 a sample-size artifact?** No. With a uniform router and 16 384 draws into 512 bins, the probability a given bin is empty is `(511/512)^16384 = e^-32.0 ≈ 1.2e-14`; expected empty bins ≈ 6e-12. A 16k-assignment histogram would light **512/512** under a uniform router. Even with 10× correlation between adjacent tokens the conclusion holds. So **80 is real concentration, not small-sample noise.** (Confidence: high. This is the coupon-collector calculation, not a paper claim.)

**0.3 Does the sign-integrator's behaviour explain "LFB did not fix it"?** Suppose hard counts and a correctly applied bias. A fully dead expert has c=0, so `sign(mean − c) = +1` every step, so `b_i` grows by 0.01 per step without bound. After 150k steps that is +1500; after a single 30k-step segment between resumes it is +300. **An additive bias of that size cannot fail to win top-k selection.** Therefore at least one of these is true:

- (a) the bias is not in the selection path (applied to the gate weights, or after top-k);
- (b) the bias is clipped, decayed, or recomputed rather than accumulated;
- (c) the counts fed to the integrator are not the assignment counts (e.g. normalized per sequence, or counted before a dropless mask);
- (d) **the histogram measures weighted gate mass, not hard assignments** — in which case LFB is doing exactly what it was designed to do and the metric is measuring something else.

(d) is the most likely, because it is the one failure mode the LFB paper explicitly creates: "It is not added to the `g_{i,t}` that weights the output of the selected experts" (arXiv:2408.15664, §4). Under (d), hard counts can be near-uniform (each expert ~0.195%) while mass stays at 8.3% top-1. That is not gradient starvation of the tail — the tail is *selected and ignored*, i.e. paid-for compute producing ~0 output.

**0.4 Resume resets are probably not the cause.** If only the counters reset but `b_i` is restored, nothing happens. If `b_i` itself resets, recovery takes ~1/γ steps: 100 steps at γ=0.01, 1000 at γ=0.001 — negligible against 30k-step segments. Resets matter only through the nonlinear path in §0.5.

**0.5 Token budget.** If batch = 8k tokens, 150k steps = **1.23B bytes seen**. Against 199M params that is **6.2 bytes/param** (or ~1.2 BPE-equivalent tokens/param at ~4.5 bytes/token), versus the ~20 tokens/param compute-optimal point (Hoffmann et al., arXiv:2203.15556, 2022-03-29). Even if batch is 8× larger, this is a data-limited run. **A 512-expert bank cannot specialize on ~4.8M tokens per expert** (1.23B × 2 / 512), and only ~31M tokens per expert among the 80 that are alive. This is an assumption about batch size — verify it. (Confidence: high on the arithmetic, medium on the batch-size premise.)

---

## 1. What actually causes expert collapse

**1.1 Rich-get-richer is documented in words, not in numbers.** Shazeer et al. state it plainly: "the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network" (arXiv:1701.06538, 2017-01-20, §4). **Measured in the paper:** no number is attached to the amplification rate; the section is a motivation for the auxiliary loss. Every later paper I found cites this sentence rather than quantifying it. **So: mechanism accepted, magnitude unquantified in the literature.** (Confidence: high that the mechanism is real; the absence of a measured amplification factor is a genuine gap.)

**1.2 There is a second, better-quantified mechanism: early routing lock-in.** OpenMoE compares routing decisions across intermediate checkpoints of a 650M–34B family and finds "the expert preferences are almost totally overlapped for different checkpoints, which means that the model has started to fix its routing at the very early stage of training. Even if we change the training data mixture (from 52.25% code to 20% code) and training objective (from UL2 to CasualLM), the routing decision is still fixed" (arXiv:2402.01739, 2024-02-01, §5.2). They also find routing is mostly a function of token ID: "routing decisions in MoE models are predominantly based on token IDs, with minimal context relevance" (§1). **Measured:** checkpoint-overlap of routing, plus robustness to a 32-point data-mixture change. **Not measured:** a live-expert fraction over steps.

This is the single most useful mechanism for your case: if routing freezes early, then whatever the router happened to do in the first few thousand steps is what you have at 150k, and no late-stage bias can restructure it — it can only shuffle which frozen experts win.

**1.3 Representation collapse is the third documented mechanism.** Chi et al. show token hidden states cluster around expert centroids, which drives routing to a few experts and is *caused by* the routing objective itself; they fix it by routing on a low-dimensional L2-normalized hypersphere with `d_e = N/2` (arXiv:2204.09179, 2022-04-20, §2.2, §4.1). **Measured:** routing consistency and gains on 7 multilingual benchmarks; no live-expert count. This is the theoretical basis for "the router's own objective collapses the representation it routes on."

**1.4 Auxiliary losses make experts *more* redundant, not less.** Two independent results:
- The aux loss produces "expert overlap and overly uniform routing, which hinders expert specialization"; the fix (orthogonality + variance losses) improves MoE baselines "by up to 23.79%" (arXiv:2505.22323, 2025-05-28, abstract).
- SimBal argues a uniform target makes the model "spend its capacity acquiring the same knowledge across multiple experts" and measures inter-expert redundancy with PES (cosine similarity of expert outputs); their no-balancing baseline collapses to a subset, while LBL keeps 100% alive but with higher redundancy (arXiv:2506.14038, 2025-06-16, §1, §4.3, Fig. 3).

**So "balance" and "specialization" are different objectives and the literature says forcing the first can damage the second.** This matters for your recipe: if your 80/512 is mass-based, adding more balancing pressure is the wrong lever.

**1.5 Is `share_u` a known aggravator?** **Not directly studied. I found no paper that tests a shared low-rank input basis across experts.** The closest evidence:
- Experts are already highly redundant even with fully independent FFNs: Mixtral-8x7B shows expert activations "close to the uniform routing baseline for all layers and domains" and little domain specialization (OLMoE, arXiv:2409.02060, 2024-09, §5, Fig. 22), and SimBal measures non-trivial PES between independent experts.
- DeepSeekMoE goes the other direction — it *isolates* shared experts so that routed experts can specialize (arXiv:2401.06066, 2024-01-11, §2). A shared expert absorbs common structure; your shared *input basis* forces every expert's representation to live in the same 64-dim subspace, which is the opposite of isolation.

**SPECULATION (my inference, not a paper claim):** with `share_u`, expert *i* computes `W_i (U x)`, so the only per-expert degree of freedom is the output map, and all experts receive gradients through the same `U`. Functional redundancy is therefore structurally likely, and redundant experts are interchangeable to the router — which makes the early tie-break (1.2) persist. That predicts high PES between experts. **Test:** compute pairwise cosine similarity of expert outputs on held-out bytes, exactly SimBal's PES (arXiv:2506.14038, §4.3). PES > 0.9 means you have replicas and the bank is wasted. This is cheap and decisive. Confidence in the mechanism: low-medium. Confidence that PES is the right measurement: high.

---

## 2. Which balancing mechanism is strongest at your scale

Read the "measured at" column before the "key numbers" column — almost nothing is measured at 200M/E=512.

| Method | Source / date | What was measured, at what scale | Key numbers (measured) | Cost: quality / throughput | Fit for E=512, top-2, dropless, byte |
|---|---|---|---|---|---|
| Switch aux loss | 2101.03961, 2021-01-11; weight 0.01 as restated by OLMoE 2409.02060 §4.1.6 | T5-Base→XXL, 2048 experts in the 1.6T model; top-1 | α=0.01 "large enough to ensure load balancing while small enough not to overwhelm the primary objective" | Interference gradients; ~0 throughput cost (one batch reduction) | Cheap, but at 512 experts and 32 tokens/expert it is a very noisy gradient; OLMoE needed 0.01 to keep 64 experts alive |
| GShard aux loss + capacity | 2006.16668, 2020-06-30; CF from ST-MoE 2202.08906 §3 | 600B params, 2048 experts, top-2, 2048 TPUv3 | capacity factor 1.25 train / 2.0 eval (ST-MoE, 32 experts) | Drops tokens at capacity; a token-choice router overflows 20–40% on some experts (Expert Choice, 2202.09368, §3.1.2) | **Incompatible with dropless** — the whole point is dropping |
| LFB, sign update | 2408.15664, 2024-08-28 | 1B/100B tokens and 3B/200B tokens, DeepSeekMoE backbone, sigmoid gate | MaxVio_global **0.72 → 0.04** (1B), **0.52 → 0.04** (3B); PPL **9.56 → 9.50** (1B), 7.97 → 7.92 (3B); best update rate **u=0.001** | None on quality; zero throughput cost | Cheapest correct thing — but the counts it integrates must be hard assignments, and your per-expert count is 32 |
| LFB, proportional `b += u·e_i` | 2408.15664, Table 3 | Same 1B run, tuned rates | MaxVio **0.044 → 0.028–0.040**; PPL **9.50 → 9.51–9.53** (worse) | Slightly better balance, slightly worse PPL | Not obviously better; note it is *absolute* error, which at 32 tokens/expert is ~1000× larger than at 1B scale |
| DeepSeek-V3 (LFB + seq-wise aux) | 2412.19437, 2024-12-27, §4.1.2, §5.3 | 671B total / 37B active, 256 routed + 1 shared, top-8, 14.8T tokens | γ=**0.001** for 14.3T tokens then **0.0**; seq-wise balance loss α=**0.0001** | α=1e-4 is "extremely small"; throughput negligible | **The reference recipe**: LFB alone was judged insufficient at sequence level even at 671B |
| Expert Choice | 2202.09368, 2022-02-18 | 16–128 experts, 100M-param experts, encoder-decoder | **>2× faster convergence** vs GShard top-2; EC-CF2 beats ST top-1 / GS top-2 on 11/11 GLUE+SuperGLUE; best config 100M/32E (128E better only for Switch top-1) | Perfect batch balance, but **causal leakage is proven** (LFB appendix D, 2408.15664) and tokens get variable expert counts | **Reject for a causal byte LM.** Not dropless-compatible either |
| Hash routing | 2106.04426, 2021-06-08; scaling law in 2202.01169 §3.4 | T5-scale LM + dialogue; compared against Switch and BASE | "either outperforms or is competitive with learning-to-route MoE… no routing parameters, no extra terms, no assignment algorithm"; hash is fit by the *same* scaling law as Sinkhorn-BASE and RL routing (2202.01169, 2022-02-02, §1) | Zero balancing cost, zero router cost; loses all context sensitivity | At 256 byte values, hashing on the byte is a fixed 256-way partition — useless as a model, **excellent as a diagnostic**: if it matches your learned router, the router is adding nothing |
| BASE (linear assignment) | 2103.16716, 2021-03-30 | 128 experts, 4 layers, WMT EN-DE | Optimal assignment, each expert gets an equal number of tokens; no aux loss, no new hyperparameters | Hungarian/auction cost per batch; batch-level assignment ⇒ same leakage class as EC | Diagnostic only |
| Sinkhorn-BASE | 2202.01169, 2022-02-02, §3.2 | Up to 200B params, hundreds of experts, five orders of magnitude | Doubly-stochastic assignment; described by the same bilinear scaling law, with saturation in E | Iterative normalization, cheap-ish; batch-level | Diagnostic only |
| LPR (latent prototypes) | 2506.21328, 2025-06-26 | 0.6B models: Mixtral/DeepSeekV3/Qwen3 shapes, 128 experts, top-8, C4 | Gini **0.790 → 0.036**, min–max ratio **6.41e-9 → 0.724**; average 0.70 → 0.035 | +prototype encoder + KL/align/diversity losses; test loss 3.673 → 3.720 (+1.3%) on DeepSeek-shape | Best balance numbers in the literature; **no evidence below 0.6B**, and it optimizes batch-marginal balance, same blind spot as LFB |
| SimBal (orthonormal router) | 2506.14038, 2025-06-16 | **230M active / 627M total**, 64 experts, top-? ; 19.9B tokens | LF alone: SEU **0.505** (softmax) / **0.381** (sigmoid). LF+LBL: SEU **0.997**, PPL 13.708→**14.154 (+3.3%)**. LF+SimBal: SEU **0.755** / 0.476, PPL **13.695** (no loss). **36% fewer tokens** to target loss | Gram-matrix regularizer + orthogonal init; no extra forward pass | **The only method measured at your scale with a utilization metric and a quality number.** Best evidence-per-cost |

### 2.1 What this table actually says

- **Nothing here is measured at E=512 with 32 tokens/expert.** The closest scale is SimBal's 230M active / 64 experts. Everything at 512+ experts is either ≥1B params with ≥100B tokens (LFB, V3) or inference-time analysis (LPR). **Extrapolation is unavoidable; say so in any writeup.**
- **LFB's own ablation is the only direct sign-vs-proportional comparison** and it favours sign on quality (9.50 vs 9.51–9.53) while proportional wins on balance (0.028 vs 0.044). The effect sizes are small *at 1B with ~100k tokens per expert per batch*. At 32 tokens/expert the variance of `e_i` is ~18% of its mean, which is the regime where a sign integrator is least informative and an unnormalized proportional rule is most explosive.
- **DeepSeek-V3 does not rely on LFB alone.** Even at 671B they add a sequence-wise balance loss at α=1e-4, and they anneal γ to 0 for the last 500B tokens. Your implementation is missing both halves of that recipe. This is the strongest single citation for "LFB as implemented is not the finished method".
- **Expert Choice and BASE are disqualified for a causal LM** — not by opinion, by the leakage proof in LFB appendix D and by the capacity/drop mechanism.
- **Hash routing's role is diagnostic.** Its scaling law is statistically indistinguishable from learned routing at 200B params (arXiv:2202.01169, §1, §5). If a fixed partition matches your learned router, the router is not doing context work.

---

## 3. Is 15.6% alive pathological at 150k steps?

**3.1 The literature almost never reports this number.** I searched for papers that plot *live/unique experts over training* rather than a balance-loss value. Found exactly three:

| Paper | Model | Metric | Reported value |
|---|---|---|---|
| SimBal, 2506.14038, Fig. 2, Table 8 | 230M active / 627M total, 64 experts, 19.9B tokens | unique experts activated on a **77M-token** validation set | **All experts** used across the full set, for every balancing variant including no-aux and LFB. Per-sequence utilization (SEU) is where they differ: 0.381 (LF, sigmoid) to 0.997 (LF+LBL) |
| OLMoE, 2409.02060, Fig. 10 | 1.3B active / 6.9B total, 64 experts, top-8 | expert assignment over training, first MoE layer | **Without LBL: "all tokens in the first layer are assigned to the 6th expert"**, all others "dead weights". With LBL 0.01: balanced |
| OpenMoE, 2402.01739, §5.2 | 650M–34B | routing overlap across checkpoints | Routing is **fixed very early** and survives a data-mixture change; no live-expert count given |

**This is a finding in itself: at the 200M–1B scale, the field reports balance *loss values*, not live-expert counts, with SimBal the sole exception.** Your metric is more informative than most published ones — which is also why there is no baseline to compare 15.6% against.

**3.2 What the closest evidence implies.**
- SimBal at 230M active reports 100% of experts used on a 77M-token stream under *every* variant, including plain LFB. If your 80/512 is measured on a stream of that size, **you are far outside the only comparable data point** and it is pathological by that standard.
- OLMoE shows the no-balancing endpoint: total collapse to one expert. 80/512 sits between "one expert" and "all experts", which is what an *ineffective but non-zero* balancing pressure looks like.
- The three-phase result (arXiv:2604.04230, 2026-04-05) says balance is *highest* early: γ_eff rises from 14 to 36–39, peaking at **30k–40k steps**, holds through 100k–400k, then *relaxes* (27 → 9) over 400k–1.2M as experts differentiate. At 150k steps a 1B model is in the most-balanced phase of its life. **So "it will fix itself later" is contradicted by the only dynamics study available.** Caveat: measured on OLMoE-1B-7B (20 checkpoints) and OpenMoE-8B (6 checkpoints), i.e. 5–40× your size.
- The same paper offers the useful reframing: late-training imbalance can be *quality-driven*, not a failure. "The router trades balance for quality as experts differentiate." A single expert at 8.3% (21× fair share) looks more like specialization than like starvation.

**3.3 Verdict on Q3.** 15.6% is **not normal by the only comparable measurement** (SimBal: 100% alive at 230M/64 experts), and **not explained by step count** (balance peaks at 30–40k and declines after 400k, so 150k should be near-best). But two caveats keep this at medium confidence: (i) no paper measures live experts at E=512/200M, and (ii) if your histogram is gate mass, the comparison to SimBal's hard-count metric is invalid. **Confidence: medium, conditional on the metric being hard counts.**

---

## 4. Practical recipe, cheapest first

Costs are engineering effort, not GPU-hours, unless stated. Effect sizes are from the cited paper's scale — they are priors, not predictions for your run.

| # | Change | Cost | Expected effect (evidence) | Confidence |
|---|---|---|---|---|
| 0 | **Log both metrics: hard top-k counts *and* weighted gate mass, on a fixed 5–10M-byte held-out stream.** Also log `exp(H)` of the routing distribution, per-expert counts, and the distribution of `b_i` (min/max/std) and its correlation with load. | 0 — one eval pass + a few lines | Decides whether the diagnosis is "LFB broken" or "LFB working, mass collapsed". SimBal's protocol is exactly this (§4.3, Table 8). Without it every later step is a guess | — |
| 1 | **Verify the bias is added to the selection logits and is checkpointed.** Add an assertion that `b` is in the top-k input, and save/restore `b`. | ~1h | If the bias is not in the selection path, LFB is a no-op and this alone explains the result (2408.15664, §4: bias is applied "before the top-K routing decision") | High that this is worth checking |
| 2 | **Set γ=0.001 (from 0.01) and anneal to 0 in the last ~10% of steps.** | one config value | V3 uses 0.001 for 14.3T tokens then 0.0 (2412.19437, §5.3); LFB's tuned 1B value is 0.001 (Table 3). At 32 tokens/expert, 0.01 injects ~18%-noise-sized steps; 0.001 gives a random-walk σ of ~0.17 over 30k steps instead of ~1.7 | Medium-high on the direction; no direct measurement at your scale |
| 3 | **Test the byte-identity hypothesis.** Compute mutual information between the byte value and the chosen expert pair, and the same conditioned on ±8 bytes of context. Also run the PES similarity metric across experts. | 0 — forward passes on held-out bytes | If MI(byte; expert) ≈ H(expert), routing is a byte lookup and 80/512 is *expected*, not pathological (OpenMoE, 2402.01739, §1, §5.2). PES > 0.9 means the 512 experts are replicas (SimBal, §4.3) | High that this is the decisive experiment |
| 4 | **Add an LBL term at 0.01 *only if* step 0 shows hard-count collapse.** | 1 line + a coefficient sweep | At 230M active: SEU 0.381→0.997, at the price of PPL **13.618→14.015 (+2.9%, sigmoid)** and 13.708→14.154 (+3.3%, softmax) (SimBal Table 8). That +3% is the honest price of near-perfect balance at your scale | **High — this is a measured number at ~your scale** |
| 5 | **Orthonormal router init (+ SimBal's Gram regularizer if you want the full method).** | ~half a day | SimBal: 36% fewer tokens to reach the same loss (50B vs 78.6B); zero PPL cost for SEU 0.755. Orthogonal init alone moves the orthogonality metric 0.0311→2.1e-8 and they note "with appropriate initialization, the load-balancing loss converges quickly" | Medium — the gain is measured but entangled with the loss term |
| 6 | **A/B the bank: E=128, top-2, same expert size.** | one training run | Same active FLOPs, 4× fewer total params, **4× tokens per expert (32→128)**. Against it: 4× less capacity, and current tokens/param (~6 bytes/param, §0.5) says capacity is not binding | Medium; this is the experiment the scaling laws point to |
| 7 | **If 6 wins: E=64, top-2 (256 assignments/expert) or E=128, top-4 (active ratio 3.1%).** | one run | 2509.23678: G_opt≈**6.78** active experts, independent of N/N_a/D; practically optimal active ratio **5–9%** (theoretical 20–43%), fitted up to 9B params / 100B tokens / 256 experts. Your E=512, k=2 is **0.4%** active — 12–22× below the practical band | Medium-low: your 199M is below their fitted N range |

**Do not do first:** add a stronger aux loss (costs ~3% PPL at your scale per SimBal, and may be solving a metric artifact), switch to Expert Choice or BASE (causal leakage, proven), or raise top-k (4× active FLOPs for an unmeasured gain).

---

## 5. Byte-level and ~200M-parameter MoE

**5.1 Byte-level MoE is nearly unstudied.** Two papers exist and neither is strong evidence:
- **EntropyMoE** (arXiv:2608.06398, 2026-07-31, cs.AI only, no venue): byte-patch MoE on a BLT-style pipeline, **E=8, top-2**, 25 MoE layers, 1.46B active / 4.06B total. Router input is the scalar patch entropy only (400 router params vs ~4e5 for hidden-state routing). Reports lowest held-out BPB vs dense and hidden-state MoE controls — **and states that "routing concentrates on a small expert subset"**, with no fix and no balance metric. That is the closest thing to your setting in the literature, and it reports the same symptom at E=8. Treat as a weak signal: single-author-scale rigor, no venue, no code link verified.
- **Kairos** (arXiv:2610.12495, 2026-09-26): byte-level multimodal model with a sparse MoE FFN; sizes a **200M total / 25M active** target analytically and explicitly says "none of its throughput figures is measured". Only overfits a 440-byte sentence at 1–15M params. **This is not evidence about MoE quality**; cite it only as "someone else considered this design point".

**5.2 The one real byte-level insight comes from a token-level paper.** OpenMoE's context-independence result (2402.01739, §1, §5.2) has a sharper consequence at byte level than at BPE level: the routing key is drawn from a 256-symbol alphabet whose unigram is extremely skewed (space alone is ~15–18% of English bytes). A context-independent router therefore *must* produce a heavily skewed expert histogram, with one expert owning the space/`e` class at several percent. **Your top-1 = 8.3% is numerically what "one expert owns a frequent byte" looks like.** The random-bytes control supports this: uniform bytes ⇒ near-uniform expert usage (439/512), skewed bytes ⇒ concentrated usage (80/512). Both observations fall out of the same mechanism with no collapse at all.

**SPECULATION (my inference):** this is the most likely single explanation of your histogram, and it predicts that the 8.3% expert is the one whose top bytes are space/`e`/`t`. **Test:** for the top-10 experts by mass, dump the byte histogram of their assigned tokens (OpenMoE's Table 9 does exactly this). If they are byte-class experts, you do not have collapse — you have a byte lookup table with 512 slots for a 256-symbol alphabet, which is a capacity-allocation problem, not a balancing problem. Confidence: medium-low until the byte histogram is dumped, but it fits both of your measurements better than any balancing story.

**5.3 At ~200M params the closest published anchor is SimBal's MoE-M** (230M active, 627M total, 19.9B tokens, 64 experts) — and it is the *only* paper in this review that measures both utilization and quality at that scale. Everything else is ≥1B. Plan accordingly: your scale is where the evidence runs out.

---

## 6. VERIFIED (sourced) vs SPECULATION (my inference)

### VERIFIED — measured in the cited paper, with the number

| Claim | Source (ID, date, location) |
|---|---|
| Rich-get-richer is self-reinforcing; no magnitude given | 1701.06538, 2017-01-20, §4 |
| LFB: MaxVio_global 0.72→0.04 (1B), 0.52→0.04 (3B); PPL 9.56→9.50, 7.97→7.92; best u=0.001 | 2408.15664, 2024-08-28, Tables 2–3 |
| LFB sign vs proportional: MaxVio 0.044 vs 0.028–0.040; PPL 9.50 vs 9.51–9.53 | 2408.15664, Table 3 |
| LFB bias is applied before top-k and *not* to the output weights | 2408.15664, §4 |
| Expert Choice leaks future tokens; proven | 2408.15664, appendix D |
| DeepSeek-V3: γ=0.001 then 0.0; seq-wise aux α=0.0001; 256 routed + 1 shared, top-8 | 2412.19437, 2024-12-27, §4.1.2, §5.3 |
| SimBal at 230M active: SEU 0.505/0.381 (LF), 0.997 (LF+LBL, PPL +3.3%/+2.9%), 0.755/0.476 (LF+SimBal, no PPL cost); 36% fewer tokens | 2506.14038, 2025-06-16, Table 8, §6.2 |
| SimBal: all experts used across a 77M-token validation set under every variant | 2506.14038, Table 8 caption |
| OLMoE without LBL: all first-layer tokens to the 6th expert; others are "dead weights"; LBL weight 0.01 | 2409.02060, 2024-09, §4.1.6, Fig. 10 |
| OpenMoE: routing is mostly token-ID based, fixed early, survives a data-mixture change 52.25%→20% code | 2402.01739, 2024-02-01, §1, §5.1, §5.2 |
| LPR: Gini 0.790→0.036, min–max 6.41e-9→0.724; avg 0.70→0.035, at 0.6B | 2506.21328, 2025-06-26, Table 1 |
| Expert Choice: >2× faster convergence vs GShard top-2; EC-CF2 wins 11/11; capacity factor 2 | 2202.09368, 2022-02-18, §4.2, Table 3 |
| Token-choice routers overflow 20–40% on some experts | 2202.09368, §3.1.2 |
| Hash routing is competitive with Switch and BASE; no router params, no aux loss; same scaling law as Sinkhorn-BASE and RL routing | 2106.04426, 2021-06-08, abstract; 2202.01169, 2022-02-02, §1 |
| BASE: linear assignment, equal tokens per expert, no aux loss / no new hyperparameters | 2103.16716, 2021-03-30, abstract |
| Three phases: γ_eff 14→36–39 peaking at 30k–40k steps, then 27→9 over 400k–1.2M; balance is traded for quality late | 2604.04230, 2026-04-05, abstract, §8 |
| Fine-grained scaling: G=1 (expert = full FFN) "almost never optimal"; optimal G rises with FLOPs; expansion rates to 64×25M active, D=16B/65B | 2402.07871, 2024-02-12, §1, Fig. 1, Fig. 3 |
| MoE scaling law: G_opt≈6.78 active experts, independent of N, N_a, D; optimal active ratio 20–43% theoretical, 5–9% practical; fitted to 9B params / 100B tokens / 256 experts | 2509.23678, 2025-09-28, §5.1, Eq. 12 |
| PEER: with 1M experts, usage close to 1/E; query BatchNorm improves balance | 2407.04153, 2024-07-04, §3, Table 2 |
| Byte-level MoE, E=8/top-2: lowest BPB vs dense/hidden-router controls, but "routing concentrates on a small expert subset" | 2608.06398, 2026-07-31, §4–5 |
| Byte-level 200M-total/25M-active MoE design exists but has no measured throughput | 2610.12495, 2026-09-26, abstract, Discussion |
| Compute-optimal ~20 tokens/param (dense LM) | 2203.15556, 2022-03-29 |

### SPECULATION — my inference, explicitly not measured anywhere I found

1. **`share_u` aggravates collapse by making experts functionally interchangeable, so the early tie-break freezes.** No paper tests a shared input basis. Basis: 2402.01739 (early lock-in) + 2204.09179 (representation collapse) + the observation that redundancy exists even without weight sharing (2409.02060, 2506.14038). Confidence: low-medium. Falsifier: PES between experts.
2. **At 32 tokens/expert, γ=0.01 makes the sign-integrator integrate noise.** The arithmetic is mine (§0.3); the noise-vs-step-size argument is standard stochastic approximation, not from any MoE paper. Confidence: medium.
3. **The 8.3% top expert is a byte-class expert (`e`/space), not a collapsed attractor.** Fits both histograms; no paper measures byte-level expert byte-histograms. Confidence: medium-low. Falsifier: per-expert byte histogram.
4. **LFB is functioning as designed and the histogram measures gate mass.** This is the only hypothesis that reconciles "LFB didn't fix it" with an unbounded sign-integrator (§0.3). Confidence: medium. Falsifier: hard-count histogram.
5. **At ~6 bytes/param, capacity is not binding, so E=128 loses little.** Chinchilla's 20 tokens/param is for BPE token LMs; the byte conversion is approximate. Confidence: low-medium. Falsifier: the E=128 A/B.
6. **E=512/top-2 is ~12–22× below the empirically optimal active ratio.** 2509.23678's fitted range starts at 20B params; 199M is extrapolation. Confidence: low-medium on the magnitude, medium on the direction.

### Cannot answer from available information

- **No paper reports live-expert fraction over training for E≥256 at ≤1B params.** SimBal is the only live-expert-over-training curve found, at 64 experts / 230M active.
- **No paper reports a quality cost for LFB at 200M params.** All LFB measurements are 1B and 3B with ≥100B tokens.
- **No paper isolates a "shared low-rank input basis" in MoE.** The nearest construct (shared experts, 2401.06066) is architecturally the opposite.
- **No paper reports byte-level (256-symbol) MoE routing statistics beyond EntropyMoE's qualitative "concentrates on a small subset".**

---

## 7. Open questions worth one run each

1. Hard-count vs gate-mass histogram — does the 80/512 survive?
2. MI(byte; expert) — is the router a byte lookup?
3. PES between experts — are the 512 experts replicas?
4. Does `b_i` actually reach large positive values for the dead experts? (If yes and they are still dead, the bias is not in the selection path.)
5. E=128/top-2 A/B at matched tokens and matched active FLOPs — what is the BPB delta?
6. What is the actual batch size in tokens? Every arithmetic conclusion in §0.5 depends on it.

---

## 8. Sources

| ID | Date | Title | Why it is here |
|---|---|---|---|
| 1701.06538 | 2017-01-20 | Outrageously Large Neural Networks (sparsely-gated MoE) | origin of rich-get-richer and the aux loss |
| 2006.16668 | 2020-06-30 | GShard | aux loss + capacity factor at 600B |
| 2101.03961 | 2021-01-11 | Switch Transformer | aux loss at 2048 experts |
| 2103.16716 | 2021-03-30 | BASE Layers | assignment-based balance, no aux loss |
| 2106.04426 | 2021-06-08 | Hash Layers | fixed routing competitive with learned |
| 2202.01169 | 2022-02-02 | Unified Scaling Laws for Routed Language Models | E-saturation; hash/Sinkhorn/RL share one law |
| 2202.08906 | 2022-02-17 | ST-MoE | router z-loss; 32 experts; CF 1.25/2.0 |
| 2202.09368 | 2022-02-18 | Expert Choice Routing | 2× convergence; overflow 20–40% |
| 2203.15556 | 2022-03-29 | Training Compute-Optimal LLMs | 20 tokens/param |
| 2204.09179 | 2022-04-20 | Representation Collapse of Sparse MoE | mechanism + hyperspherical routing |
| 2401.06066 | 2024-01-11 | DeepSeekMoE | fine-grained + shared experts; routing collapse framing |
| 2402.01739 | 2024-02-01 | OpenMoE | token-ID routing, early lock-in, dead experts |
| 2402.07871 | 2024-02-12 | Scaling Laws for Fine-Grained MoE | granularity; G=1 suboptimal |
| 2407.04153 | 2024-07-04 | Mixture of A Million Experts (PEER) | 1M experts, near-uniform usage |
| 2408.15664 | 2024-08-28 | Auxiliary-Loss-Free Load Balancing | sign vs proportional; MaxVio numbers; leakage proof |
| 2409.02060 | 2024-09 | OLMoE | dead weights without LBL; 64 experts |
| 2412.19437 | 2024-12-27 | DeepSeek-V3 | γ=0.001→0.0; α=0.0001 seq-wise |
| 2505.22323 | 2025-05-28 | Advancing Expert Specialization | aux loss causes expert overlap |
| 2506.14038 | 2025-06-16 | SimBal: Similarity-Preserving Routers | **the scale-matched anchor**: SEU, PPL cost, live experts |
| 2506.21328 | 2025-06-26 | Latent Prototype Routing | Gini 0.79→0.036, min–max 6e-9→0.72 |
| 2509.23678 | 2025-09-28 | Comprehensive MoE Scaling Law | G_opt≈6.78; active ratio 5–9% practical |
| 2604.04230 | 2026-04-05 | Three Phases of Expert Routing | balance is non-monotone; peaks at 30–40k steps |
| 2608.06398 | 2026-07-31 | EntropyMoE | byte-patch MoE, E=8, concentrates anyway |
| 2610.12495 | 2026-09-26 | Kairos | byte-level 200M/25M MoE design, unmeasured |
