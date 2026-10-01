# What goes on top of random-depth training in dormouse — adaptive depth without a collapse channel (2026-09-27)

**Question.** dormouse runs ONE shared looped block at `max_iter=4`. PonderNet halting was deleted
(ADR-0013) after two measured collapses, and `--rand-depth` (uniform T ∈ 1..4 per step, both
averages divided by T) shipped as the cheap replacement. The owner wants adaptive depth **in both
directions**. This report decides what, if anything, is added on top, and what the A/B against
fixed-depth-4 must measure.

**Method.** Primary sources only (arXiv full texts, opened and quoted below; official repos where they
exist). 6 rounds: (1) stochastic-depth lineage + Huginn, (2) MoR, (3) arXiv API scans of 2026,
(4) the elastic-depth family (LoopFormer, T-LoopFormer), (5) the early-exit family (CALM, LayerSkip,
diminishing-returns study), (6) the depth-dynamics family (STARS, Think-Shallow-Solve-Deep, 15M GDN
recurrence, adaptive-depth-use study, scaling-exponent study). Repo code was read to fix the
architecture facts every design depends on. Anything I could not open or that no source measures is
marked **NOT VERIFIED**.

**Verdict in one line.** Nothing learned should be added yet: measure the depth curve and the
free oracle from one checkpoint first — the literature's best measured adaptive-depth gain in a
looped LM is +2.2 points at 1.7B and it comes from 2.7% of inputs, which at 7.5M byte level is
almost certainly below our noise floor, so the honest deliverable is a **measured compute/quality
frontier with a confidence exit**, not a router.

---

## 0. The dormouse facts that constrain every design (read from the code, not assumed)

* The loop does **not** accumulate a residual stream across iterations: `loop_block.rs:381` resets
  `h = h_ctx` after each iteration, so the state that carries across iterations is the attention
  memory (KDA state + MSA KV) and the per-iteration write reaches the loss only through that
  iteration's readout. There is a learned per-iteration embedding (`iter_embed` row `iter`) added to
  the block input, i.e. discrete "time" conditioning — this is the same idea as LoopFormer's
  time/step conditioning, and it is already in the architecture.
* The readout is the **mean of the per-iteration outputs**: `out_acc = Σ_n step_out_n / T`, then
  `norm → lm_head` (`model.rs:190-198`). The per-iteration logits are trained too: `rec` is the mean
  of `CE_n` over the *executed* iterations (`loop_block.rs:361-377`).
* `--rand-depth` samples T per step and divides both averages by what ran, so a depth-T step is an
  honest depth-T model. Under this arm, **the prefix mean over 1..T is a trained readout at every
  T** — that fact is the whole basis of §3.
* Held-out eval: a fixed rewound window, `eval_batches=20` × `batch×seq_len` = 20 × 5 120 B =
  **102 400 B** (and NOT a fixed 100 KB: the window is
  `eval_batches × batch × seq_len`, so it is 102 400 B at batch 10 and 20 480 B at
  batch 2 - two runs at different batch sizes scored different amounts of text;
  the byte count on the eval line is the authority), scored as the CE of the
  *final* readout only. It calls
  `forward_with_hidden(..., targets=None, ...)`, so **no per-iteration CE is computed at eval**, and
  there is **no flag to force the eval depth** (`set_depth` exists in core, nothing exposes it).
* Repo doctrine: steps are launch-bound, so training-side FLOP savings from
  adaptive depth buy ~no wall-clock. The ROI of adaptive depth here is inference-side plus
  depth-robustness, not throughput.
  **CORRECTED 2026-09-29: this doctrine line now has an instrument behind it,
  and it is confirmed.** It was originally "launch-bound (fixed ≈465 ms/step)";
  the 465 ms was fitted to step-0 readings and is struck (`benches/history.tsv`,
  2026-09-29 - a warm step at batch 8 is 244 ms, a step-0 step is 5549 ms, 23x
  apart, because the cubecl autotune cache is cold for the first steps; `--timers`
  used to print on `step % 50 == 0` only, and that defect is now fixed in
  `b8a47ee` - it follows `--log-every`). What replaces it is direct: over a 150-step
  warm run at batch 32 the GPU is **13.3 % utilised on average, 142 of 180 samples
  at <=5 %**. Too many small kernels to fill the SMs. The conclusion this document
  rests on - FLOP savings buy ~no wall clock at our scale - is therefore stronger
  than when it was written, and the 465 ms number should not be quoted.

---

## 1. Random-depth / stochastic-depth training: does one checkpoint serve ALL depths?

### 1.1 The classic lineage says "shallow during training, full at test", and it re-weights by survival probability

Huang et al. 2016 [R1] is the origin and two of its details matter to us:

* The framing is literally ours: *"a network with a small expected depth during training, but a large
  depth during testing"*; they train short and test deep, reaching **>1200 layers** and 4.91%
  CIFAR-10 error, 5.25% vs 6.41% for the constant-depth ResNet.
* The survival probability **decays with depth**: `p_ℓ = 1 − (ℓ/L)(1 − p_L)` with `p_0 = 1`,
  `p_L = 0.5` by default. Their stated reason — *"earlier layers extract low-level features that will
  be used by later layers and should therefore be more reliably present"* — and they empirically
  compare uniform vs decaying and keep decaying. In loop language: **sample the iteration count biased
  toward the early iterations.**
* At test time they **re-weight each unit by its survival probability** (`H_ℓ^Test = ReLU(p_ℓ f_ℓ(H)
  + H)`), because each unit was only active for a fraction `p_ℓ` of updates. dormouse's "divide the
  readout average and the CE by T" is the same correction in a different coordinate system: it makes
  the truncated run's readout the readout of a depth-T model rather than a partial sum of a depth-4
  one. That part of the shipped design has 10-year-old precedent.
* They also read the mechanism as an **implicit ensemble of 2^L subnetworks** — the reason a
  depth-robust model is a better model, not merely a cheaper one.

### 1.2 The looped-model evidence: yes, one checkpoint serves every depth in the trained range

* **Huginn** (Geiping et al. 2502.05171v2, 3.5B / 800B tokens) [R2] samples the loop count from a
  **log-normal Poisson** (mean `r̄ = 32`, σ = 0.5), backprops through only the last **k = 8**
  iterations, and minimizes `L(θ) = E_x E_{r~Λ} L(m_θ(x, r), x′)` — **the loss at the sampled depth,
  with no depth mixture and no halting weights**. The depth-robustness claim is Figure 6 (right):
  *"Plot of val ppl at recurrent depths 1, 4, 8, 16, 32, 64. During training, the model improves in
  perplexity on all levels of recurrence."* One checkpoint, depths 1→64, all improving. Two
  cautions: their "Bad Run 1" stalled because hidden-state correlation went to 1.0 (representation
  collapse) — initialization and architecture discipline are load-bearing for a recurrent run; and the
  distribution *"most often samples values less than r̄ … contains a heavy tail"* — shallow-biased,
  not uniform.
* **LoopFormer** (Jedi et al. 2602.11451) [R3] makes the budget a first-class input: every batch runs
  the full L-loop trajectory **and** a sampled shortcut `S ~ U{1,…,L−1}`, with
  `L = L_L + 0.1·L_S + 0.1·L_cons`, where `L_cons` is a **stop-gradient** per-token-logit consistency
  loss pulling the short trajectory onto the long one. Each loop is conditioned on normalized time
  `t` and step size `Δ`. Result: *"at small compute (few loops), LoopFormer is on the par with the
  vanilla baseline, while increasing the loop budget leads to consistent improvements"* — the
  elastic-depth curve. Cost of the recipe ≈ `C(L) + E[C(M)]` ≈ 1.5× the fixed-depth step. They also
  report that looped models trail non-looped ones on perplexity and that the **step schedule matters**
  (best schedules give larger steps early, finer late; PPL spread across schedules up to ~3).
* **T-LoopFormer** (Yu et al. 2609.15160, d=2048, 1 block × up to 8 loops, 100B FineWeb-Edu tokens)
  [R4] — see §5; its loop histogram at 24× FLOPs is 0.23 / 36.85 / 23.73 / 8.14 / 13.87 / 2.16 /
  11.26 / 3.77 % for 1..8 loops (mean **3.69** of 8), i.e. heavily shallow-skewed with a tail, and it
  notes *"either excessive or insufficient elastic-depth unrolling may lead to incorrect token
  outputs"* and that at reduced budgets (12×/6×) it **loses to the non-looped base** because hard
  tokens are under-iterated.
* **Where random depth stops helping.** STARS (2605.26733) [R5] tested exactly our sampler at small
  scale (d=512, 8 heads, `T_train = 4`, test-time sweep) and reports that *"common remedies, such as
  auxiliary Prelude/Coda layers, L2 regularization, or random loop sampling, cannot totally resolve
  this deadlock"*. Random depth buys depth-robustness; it does **not** make the loop *settle*.
* **Extrapolation ceiling.** Retrofitting recurrent depth (2608.11233) [R6] reports the operation
  extrapolates *"to roughly 1.5 times its supervised depth, holding 70% accuracy through depth 18"*,
  led through depth 11 and trailed beyond. So: within the trained range, random depth gives you all
  depths; beyond ~1.5× it degrades. dormouse's max_iter=4 is a trained depth, so depths 1..4 are in
  range by construction.

### 1.3 Answer to Q1

* **Depth-robustness claim: supported.** Random-depth training yields one checkpoint usable at every
  depth in the trained range (Huginn Fig 6; LoopFormer Table 2; T-LoopFormer Table 4). Our shipped
  uniform sampler is the conservative end of the family (LoopFormer and T-LoopFormer both use
  uniform; Huginn and Huang both argue for shallow bias).
* **Which schedule:** no source shows a within-training *curriculum* (e.g. grow T over steps) beating
  uniform sampling for a looped LM. The two changes that are backed are (a) **shallow bias with a
  heavy tail** (Huang's decaying survival, Huginn's log-normal-Poisson) and (b) **always also train
  the deepest trajectory** (LoopFormer's dual-trajectory objective). 2609.19107 [R7] is the one
  2026 paper arguing a *growth* curriculum ("it is compute-optimal to increase the number of loops
  with scale") — but at 7.4B, in a data-constrained multi-epoch regime, which is not ours.
  **Recommendation: keep uniform. If we change anything, change it to LoopFormer's dual-trajectory
  form (deep always + sampled shortcut), not to a curriculum.**
* **NOT VERIFIED:** no source measures random-depth training for a *byte-level* 256-vocab model, and
  no source measures it at ≤10M parameters on natural language.

---

## 2. MoR router stability: the precise mechanism, and a correction to ADR-0013's reason

MoR (Bae et al. 2507.10524v3, NeurIPS 2025) [R8], N_r = 3 recursions of a shared Middle-Cycle block,
135M–1.7B.

### 2.1 The mechanism, exactly

* **Expert-choice** (their best): at recursion r, a scalar score per token; exactly **k tokens pass**
  (k set by a β-percentile capacity over the batch's scores), the rest pass through unchanged. Because
  the top-k is **rank-based and capacity-fixed, exactly k tokens always get through** regardless of
  what the scores are numerically. There is no global mass the optimizer can shrink.
* **The anchor that keeps the scores calibrated**: expert-choice top-k is non-causal, so they patch
  causality with a binary loss. The winning variant is the **auxiliary loss on the main router
  itself**: *"applies the binary cross-entropy loss to the main router itself, enabling it to
  simultaneously learn to push top-k tokens towards one and others towards zero"*, with the main
  router's own top-k membership as the target (selected → 1, unselected → 0), plain **Linear**
  router, **sigmoid**, output scaling α = 0.1, coefficient 0.001, z-loss optional. Measured effect:
  *"In all steps, auxiliary loss achieves a perfect separation in router outputs, with selected
  tokens sharply concentrated near a routing score of 1.0 and unselected tokens clustering near
  0.0"*, dead-token ratio 0.1%. The separate grad-blocked aux *router* was worse and its bad variants
  hit **66.7% dead tokens** (positional bias).
* **No depth-0 mode:** every token traverses recursion 1, so "route everything to no compute" is not
  in the hypothesis space.
* **No loss mixture:** the LM loss is computed once, at each token's final depth. This is the channel
  our collapse used (a p-weighted mixture is a knob that multiplies the loss).

### 2.2 Is MoR's small-scale deficit evidence against routing? Partly — and ADR-0013's reason is wrong

The claim in ADR-0013 ("MoR underperforms vanilla at 135M (we are 7.5M)") quotes the paper correctly
(*"it underperforms the vanilla model at the smallest model size (135M) — likely due to a recursive
capacity bottleneck"*), but the isoFLOP table separates the two causes [R8, Table 9]:

| 135M base, N_r=3, 2e18 FLOPs | FineWeb NLL ↓ |
|---|---|
| Vanilla (106M non-emb, 30 layers) | **3.0922** |
| Recursive (42M unique, 1+10+1) | 3.2058 |
| MoR (42M unique, 1+10+1) | **3.1077** |

At 5e18: 2.9464 / 3.0534 / 3.0192. So **MoR beats the same-architecture recursive baseline at every
scale including 135M**; the deficit is against a 2.5×-larger *unshared* transformer, and the paper
attributes it to the **recursion (weight sharing), not the router**. That distinction matters for
dormouse: our fixed-depth-4 arm *already pays* the recursion bottleneck, so a router is not facing the
same handicap the paper's 135M vanilla row faced. It does **not** change the verdict (see below) but it
changes the *reason*, and the reason was written into an ADR.

### 2.3 Would a MoR-lite inherit the stability?

Only if the ranking + anchoring are kept. The anti-collapse properties are structural, not
incidental: fixed capacity (rank-based, always fills), a binary target recomputed per batch (a
self-supervised classifier, not a preference over loss mass), a floor of one recursion, and an
unweighted LM loss. A "MoR-lite" that trains a router end-to-end with no aux BCE reintroduces the
scale channel the aux loss closes — the router can learn to score everything low, capacity still
fills the top-k, so it degrades to *arbitrary* selection rather than collapse, which is milder but
still a silent bug (MoR's own failure catalogue: dead tokens, MaxVio, NaN rows).

### 2.4 The blocker is our stack, not the stability

MoR's own efficiency story needs per-depth **token-subset gather/scatter** on hidden states (their
training gather reduces memory 21.9%, inference latency 26.4%; TPOT 0.197 s vs 0.389 s for the
non-routed looped baseline) or recursion-wise KV. Per-depth gathers on token subsets are dynamic
indexing of the tensor kind that cubecl crashes on (sm_120 `CUDA_ERROR_ILLEGAL_ADDRESS`, repo
doctrine). At max_iter=4 with byte vocab 256 the FLOP saving is ~3 loop iterations of a 7.5M model —
worth nothing against a launch-bound step. **Verdict: MoR stays rejected on stack grounds (as
ADR-0013 says), but for the gather reason, not the 135M reason.**

---

## 3. Confidence-based early exit: the criterion, the cost, and the dormouse-specific exit point

### 3.1 What the criterion should be

* **CALM** (Schuster et al. 2207.07061v2, NeurIPS 2022 oral, T5-1.1 8-layer, code in t5x) [R9]:
  exit when `c_t^i ≥ λ_t^i`. Three measures compared; the winner is the **softmax response =
  difference between the top two softmax values** (a margin), not the top-1 probability. They
  **share the output embedding and one early-exit classifier across all layers** — no per-layer
  heads. Headline: up to **3× speedup** with provable consistency; the *oracle* bound is 5.2×.
* **A single threshold is a cliff.** CALM's decaying threshold
  `λ'(λ,t) = clip(0.9λ + 0.1·e^{−τt/N})` (τ = 4) exists because with τ=0 *"attempting to improve the
  efficiency will lead to a drastic drop of more than 10 points in the textual similarity against the
  full model's prediction"*. With depth ≤ 4 we have three decision points, so the analog is a
  monotone threshold schedule (hardest to exit at iteration 1, easiest at 3), not one τ.
* **Missing states.** CALM §3.3.1: copying the *last computed* hidden state forward is robust;
  copying K/V from lower layers into skipped layers collapses to 23.02 ROUGE-L. This is the
  dormouse analogue of "an early-exited predecessor has no K/V at loops 2..4".
* **LayerSkip** (Elhoushi et al. 2404.16710v2, Meta) [R10]: make early exits *accurate* at training
  time with **layer dropout (rate increasing with depth) + an early-exit loss where all layers share
  the same exit**; 1.34–2.16× speedups (2.16× CNN/DM, 1.82× code, 2.0× TOPv2), plus
  self-speculative decoding. Dormouse already has the structural equivalent: every iteration shares
  `out_proj` + `lm_head` and each iteration's own logits are in the loss.
* **2026 successor, and a warning.** "The Diminishing Returns of Early-Exit Decoding in Modern LLMs"
  (2603.23701) [R11] defines the metric we should borrow: skip ratio `w_ℓ = (L−ℓ)/L`, layer-to-final
  cosine similarity `S_ℓ`, and a scalarized **EAS** (weighted geometric mean). Findings:
  early-exit effectiveness **decreases across newer model generations**, is **higher for larger
  models (notably >20B)**, **higher for dense than MoE/SSM**, and **higher for base pretrained models
  than for tuned ones**. That is the honest prior for us: a 7.5M base model is the *favourable* end on
  architecture and training recipe and the *unfavourable* end on scale.

### 3.2 The controlled looped-LM comparison (the number that actually calibrates us)

RecurTrace (2609.03379v2) Table 2 [R12], 1.7B, MathQA, 2400 items, 3 training seeds × 8
non-overlapping eval seeds, all on the same looped backbone:

| method | acc % | mean loops | Δ correct vs fixed T=2 |
|---|---|---|---|
| fixed T=1 | 49.96 | 1.00 | −114 |
| **fixed T=2** | **54.71** | 2.00 | 0 |
| fixed T=8 | 53.62 | 8.00 | −26 |
| fixed T=16 | 47.54 | 16.00 | −172 |
| ACT | 49.96 | 1.00 | −114 (collapse) |
| PonderNet | 49.96 | 1.00 | −114 (collapse) |
| ACT+floor / PonderNet+floor | 54.71 | 2.00 | 0 (exactly fixed depth) |
| CALM (top-prob) | 53.04 | 4.15 | −40 |
| CALM (margin) | 54.12 | 5.63 | −14 |
| LoopUS-Conf | 55.25 | 3.21 | +13 |
| TaH-Mismatch | 55.67 | 2.12 | +23 |
| RecurTrace (oracle-supervised head) | 56.92 | 2.04 | **+53** |

Read it four ways, because each way is a design rule for us:

1. **Fixed depth peaks and then falls** (54.71 at 2 → 47.54 at 16). The "fixed depth overthinks"
   half of the owner's premise is real and measured on a looped backbone.
2. **Confidence exit overspends and loses**: margin beats top-prob (as CALM says), but even the
   margin rule spends 5.63 loops to lose 14 items. CALM's failure mode is inefficiency, not
   divergence — which is why it is the safe rung.
3. **Penalty-based halting is the bug.** ACT and PonderNet hit 49.96 @ 1.00 loop — the exact dormouse
   failure, replicated on a different backbone. Adding a floor makes them *exactly* fixed depth: the
   machinery then buys nothing.
4. **The entire best measured gain comes from 2.7% of inputs.** At the operating point (τ=0.50,
   T_min=2), *"97.3% of items stop at the floor"* and *"~65 items per training seed deepen further yet
   supply the entire +53 net gain (56 wrong-to-right, 3 right-to-wrong)"*. The floor was selected so
   it coincides with the best fixed depth. So adaptive depth is not "easy tokens skip" (that is pure
   compute) — it is "a few hard items get more depth than the fixed budget gives them".

### 3.3 The dormouse-specific question: which iteration is the natural exit point?

Our readout is the **mean of per-iteration outputs**, so there are two candidates and they are not
the same function:

* **Prefix mean** `(1/k)·Σ_{n≤k} step_out_n` — exactly the readout a depth-k dormouse produces, and
  **under `--rand-depth` it is trained at every k** (the arm divides by what ran). Under fixed-depth-4
  training it was never a readout, only its per-iteration logits were trained.
* **Iteration-k logits alone** `lm_head(norm(step_out_k))` — trained at every k under *both* arms
  (per-iteration CE is in the loss), but it is *not* what the depth-4 model ships.

Consequences, all actionable:
* Confidence exit composes with `--rand-depth` for free: exit at k = read the prefix mean at k, and
  that quantity is exactly the depth-k model. No retraining, no new head, no new readout.
* **The A/B must fix the readout** to be fair: compare arms at the same k using iteration-k logits
  alone, or compare only rand-depth-trained arms on prefix means. Otherwise "rand-depth wins at depth
  2" can just mean "the rand-depth arm was trained on the depth-2 readout".
* The prefix mean is a strictly better early-exit readout than iteration-k alone if the later
  iterations are not worse than the earlier ones — and that is *exactly* the "latent overthinking"
  hypothesis (TaH) which we have not measured. Cheap to measure, see §6.
* Per-token exit inside a shared context has a real KV problem here: an early-exited predecessor has
  no K/V at loops 2..4, and the KDA state is a delta-rule state, so "copy the last computed state"
  (CALM's robust variant) is a design decision, not a freebie. **Sequence-level (per-row) exit has
  none of this** and is the right first rung.

### 3.4 The exit criterion I would ship: margin, plus a displacement check

* **Margin** `p₁ − p₂` on the fp32 logits (CALM's winner; RecurTrace: 54.12@5.63 vs 53.04@4.15 for
  top-prob). We already cast to fp32 before the head, so this is free.
* **The 2608.18222 sufficient condition** [R13] is worth more than the softmax heuristic, and it is
  cheap: let `δ_t = ‖z_{t+1} − z_t‖` be the per-step displacement and `μ(z_t)` the decision margin.
  If `R_t = Σ_{u≥t} δ_u < μ(z_t)`, *"the decoded answer cannot change under further iterations"*.
  That is a *provable* early-exit rule that does not rely on softmax being calibrated — and our loop
  hands us `h` and `h_ctx` per iteration, so `δ` is one extra reduction. Proposition 1 also names the
  opposite failure, **guess-freezing**, which is the thing a confidence rule cannot see: the answer is
  frozen early and freezing does not track difficulty.

---

## 4. Overthinking / underthinking: is the premise right, and does adaptive depth pay at small scale?

### 4.1 The premise is right (both directions), and the mechanism is known

* **Peak-then-collapse on a real looped LM**: STARS (2605.26733) Fig. 1, *"Performance of Ouro-1.4B
  on GSM8K across different recurrent steps"*: performance *"exhibits a peak at a certain iteration
  depth and deteriorates sharply or even collapses entirely as iterations further increase"*;
  quantified: **Ouro-1.4B drops 20.47% from its peak after 8 recurrent steps**; their method drops
  8.26% and improves the peak by 4.01%. Their explanation: *"direct supervised fine-tuning fails to
  equip the model with a test-time scalable reasoning capability … the model tends to overfit to the
  specific recurrent iteration during training."* Depth-specific overfitting is the disease;
  random-depth training is the partial cure.
* **The strongest looped LM, measured per token**: 2608.18222 applies its instruments to
  **Huginn-3.5B** and finds, across **2 585 per-token latent trajectories, not one settles**
  (σ_eff = 1.000 ± 0.003); the answer freezes but *"freezing does not track difficulty: guess-freezing,
  not a computation completing"*; and on a procedural single-token probe accuracy **peaks shallow and
  collapses: carry 0.69 at r=8 → 0.00 for r ≥ 32**, every input freezing onto the same wrong token.
  Across 164 competent runs, of the 17 settling runs **none** loses more than 0.004 EM to extra depth,
  while **30% of marginal and 22% of drifting runs lose more than 0.1** (up to 0.98).
* **Underthinking side**: T-LoopFormer states it symmetrically — *"either excessive or insufficient
  elastic-depth unrolling may lead to incorrect token outputs"*, and at 12×/6× budgets it loses to the
  non-looped base because hard tokens are under-iterated. 2607.10128 (ERM) concludes *"deeper
  recurrence does not necessarily improve reasoning accuracy"*.

### 4.2 Does adaptive depth pay at ≤15M? The evidence is thin and points the wrong way

* **Does depth itself pay at 15M? Yes, and specifically the mixer.** "Allocating Recurrent Compute in
  Looped LMs" (2608.18230) [R14] compares NoLoop / FullLoop / MixerLoop at **15.7M and 116.9M**
  unique params on a **Gated DeltaNet** backbone (the same family as dormouse's KDA), `T = 4`,
  52.4B tokens, matched everything: mixer-only recurrence **consistently improves over NoLoop**, and
  **MixerLoop surpasses FullLoop on aggregate CORE at 15M** while cutting recurrent-backbone projection
  FLOPs 45.9% (33.9% end-to-end) and reaching 1.52× FullLoop prefill throughput. This is the best
  small-scale evidence that *recurrent depth buys something at our scale* — and it says the value
  lives in the **state-update (memory) arm, not in re-running the dense FFN**.
* **Does *per-input adaptive depth* pay at small scale? No source says yes.** RecurTrace's gain grows
  with scale: *"the gain growing with model size from 0.6 to 3.4 points"* over **0.6B, 1.7B, 4B, 8B**
  — the smallest tested is 0.6B and the trend is monotone increasing. The other 2026 adaptive-depth
  results are all ≥0.6B: TaH fine-tunes **Qwen3-Base 0.6B and 1.7B**; CDB's 1.5–1.9× throughput is
  measured on **Ouro 1.4B and Huginn 3.5B**; T-LoopFormer is d=2048.
* **And the base-model premise is itself only weakly supported.** "Do Transformers Use their Depth
  Adaptively?" (2604.12426) [R15] runs the logit lens + causal patching over five families from
  120M to 14B: *"For pretrained models, we find some limited evidence for adaptive depth use"*;
  clearer and more consistent evidence appears only **after finetuning on the task**. Depth-adaptive
  behaviour in a *base pretrained* model — dormouse's regime — is thin evidence, and the clearest
  signals are in the largest models.
* **TaH's headroom is a percentage of tokens, not a percentage of loss.** On Ouro-1.7B the oracle
  iteration policy *"iterates on 12–19% of tokens"* and improves downstream performance by 2.0–7.3%;
  the shipped decider skips **93% of tokens** while beating always-iterate by 3.8–4.4%. Note the
  supervision design: **the decider is trained by imitation of a static oracle policy** in a second
  stage, with a depth-aware LoRA applied **only at iterations d > 1** so the first-pass prediction is
  preserved. Also note the scale: subword, 0.6B+, post-training on Open-R1.

**Verdict on Q4.** Both halves of the owner's premise are real *in the literature* (fixed depth
over- and under-shoots), but the *exploitation* of that variation is proven only at ≥0.6B and grows
with size, and the base-model evidence for adaptive depth use is explicitly "limited". At 7.5M bytes
the expected measurable accuracy gain from any router is far below what our 100 KB window can
resolve. Adaptive depth at our scale is a **compute** lever first and a **quality** lever only
conditionally.

---

## 5. 2026 fresh scan (arXiv 2602–2609): what is new and is it collapse-proof by construction

Scans run: `abs:"looped" AND abs:"depth"`, `abs:"adaptive depth" AND abs:"language model"`,
`ti:"early exit" AND abs:"language model"`, `abs:"PonderNet"`, `abs:"stochastic depth" AND
abs:"transformer"`, `abs:"recurrent depth"` (arXiv API, 30 hits each, sorted by date).

| id / date | what it is | adaptive depth? | collapse-proof by construction? | verdict for us |
|---|---|---|---|---|
| **2609.15160** T-LoopFormer (Sep 14) | 2-layer MLP router on the **initial** hidden state → softmax over M depths → **cumulative** `p^(i)=Σ_{l≥i}π_l`, token active iff `p^(i) > 0.5`; loss `L_L + 0.1 L_S + 0.1 L_align`; recursion-wise KV; gather-scatter | Yes, per-token, causal (decision uses only the token itself) | **Yes, and it is the cleanest 2026 example.** No depth loss term at all — no penalty, no prior, no loss-mass mixture. Selection is nested (`A₁ ⊇ A₂ ⊇ …`) and thresholded, so there is no scalar to shrink; depth-1 requires `π₁ > 0.5` and got only 0.23% of tokens. | Best *template* for a router's shape; the gather-scatter is the sm_120 blocker. Also documents that at reduced budgets it loses to the base — the underthinking failure. |
| **2605.26733** STARS (May 26) | Jacobian spectral-radius regularization (single-step power iteration + JVP) + random loop sampling; constrains latents to asymptotically stable fixed points | No (fixed-point training, not adaptive depth) | N/A — and it is the paper that says **random loop sampling alone does not fix stability** | **Not implementable**: JVP needs forward-mode AD, burn/cubecl is reverse-mode. Cite as the diagnosis (overthinking = spectral radius > 1), steal the cheap cousin in 2608.18222. |
| **2608.18222** Think Shallow, Solve Deep (Aug 18) | Dynamical regime taxonomy (settling / marginal / drifting), 4 behavioral criteria R1–R4, Prop. 1 depth-safety, **label-free repair**: rank-16 LoRA on the recurrent core trained to pull deep states `r~U[16,128]` toward the frozen base's own shallow state at `r=8` | It *diagnoses* useful test-time depth; the repair restores depth-**safety**, not capability | The repair is a **stop-gradient self-anchoring** term — a regression to our own shallow state, no labels, no mass knob. *"Conversion never appears"* | **The cheapest genuinely new idea for us.** See §6 rung 3. |
| **2608.18230** Allocating Recurrent Compute (Aug 18) | MixerLoop: repeat the GDN mixer, apply the dense FFN once; 15M/110M | No | N/A | Adjacent but load-bearing: tells us which part of our loop should be spending the depth. |
| **2608.09444** Continuous Depth Batching (Aug 10) | Scheduling at loop granularity, boundary stages in a separate priority queue, exit decided one step ahead; 99% of theoretical max speedup, 1.5–1.9× throughput on Ouro 1.4B / Huginn 3.5B | Systems layer | N/A | Tells us what a *correct* adaptive-depth serving stack costs, and confirms the win is throughput. Out of scope (we have no serving fleet). |
| **2609.03379** RecurTrace (Sep 3) | Oracle-distilled halting: `y_t = 1[min_{t'>t} L_t' < L_t − δ]`, BCE on `σ(w^T z_t + b)`, `z_t` = pooled block state, `T_min = 2` floor, τ picked on held-out NLL | Yes, per-sequence | Yes: target is a **measured per-batch** quantity, no compute penalty, LM loss unweighted, floor makes depth-1 unreachable | The only trained halting with 2026 evidence of not collapsing (Table 2 above). Rung 4 for us, and the oracle label is **already computed in our loop**. |
| **2603.23701** Diminishing Returns of Early Exit (Mar 24) | EAS metric: skip ratio, layer-to-final similarity, weighted geometric mean; less headroom in newer/larger-tuned models, more in base and >20B dense | Diagnostic | N/A | Borrow the metric. It is the pre-flight test for §3. |
| **2603.21365** TIDE, **2510.04871** TRM, **2607.10128** ERM, **2608.03263** ignition-at-readout, **2609.19107** scaling exponents | token-level EE / 7M recursive reasoner / energy-guided depth / readout-commitment study / model-growth exponents | mixed | — | Peripheral; ERM's *"deeper recurrence does not necessarily improve reasoning accuracy"* is the quotable line. 2608.03263's *"Intermediates were never recoverable through the tied readout (relay 0.00)"* is a **caution** for intermediate-iteration readouts, measured at 30M. |
| DeepLoop 2607.13491, SMELT 2609.01343, Hyperloop 2604.21254, Training-Free 2605.23872, LoopUS 2605.11011, Huginn 2502.05171 | the fixed-depth 2026 SOTA | **none use trained halting** | N/A | From the 2026-09-26 report; **not re-opened in this session (NOT RE-VERIFIED here)**, but consistent with everything above. |

**Answer to Q5.** Yes — one mechanism is new and structurally collapse-proof: **T-LoopFormer's
router shape** (a threshold on a cumulative softmax over depths, trained only by the LM loss of the
sampled trajectory, with an alignment term). And one mechanism is new and cheap:
**2608.18222's label-free latent anchoring** for depth-safety. Neither needs a halting preference, a
prior, or a compute penalty, which is precisely why neither can collapse into our p→0 failure.

---

## 6. For dormouse: the ranked design

Pre-registered order. Each rung states its loss, why it cannot collapse, and what kills it. Nothing
here is a halting head.

### Rung 0 — MEASURE, before writing any adaptive code (~1 day, no GPU-heavy training)

Three numbers from **one existing checkpoint**, on the existing rewound 100 KB / 20-batch window:

1. **The depth curve**: `BPB@k` for k = 1..4, same weights, same window. Needs one small change —
   a `--eval-depths 1,2,3,4` flag that loops `eval_model.set_loop_depth(Some(k))` inside the eval
   block and resets to `None` (`set_depth` already exists in `loop_block.rs:372`; nothing exposes it).
   *Also* report each k twice: prefix-mean-1..k and iteration-k logits alone (§3.3), because those are
   different functions and only the first is trained under `--rand-depth`.
2. **Per-iteration CE at eval**: the eval currently passes `targets=None`, so pass `Some(ey)` and
   return the per-iteration CE vector from the loop (~5 lines). This yields ΔCE per iteration and,
   for free, the **RecurTrace oracle label rate**: the fraction of positions where
   `min_{t'>t} L_t' < L_t − δ`, i.e. **the exact upper bound on what any adaptive scheme can win**.
3. **EAS per depth** (2603.23701): cosine similarity between the depth-k readout and the depth-4
   readout, vs skip ratio (k−1)/4. Four numbers, no training.

**Decision rule (pre-register this):**
* If `BPB@4 − BPB@3` ≤ noise and the oracle rate is small → **depth 4 is not paying; adaptive depth
  buys compute only.** Ship the inference knob, close the adaptive-depth question, and spend the
  budget on scale (the owner's 1B plan) instead. *This is the outcome I expect, and it is a legitimate
  answer.*
* If `BPB@4 − BPB@3` is real (> the noise floor below) **and** the oracle rate is non-trivial → the
  remaining rungs are worth building, in order.
* If `BPB@4 < BPB@3` (peak-then-fall) → we have reproduced latent overthinking at 7.5M; Rung 3
  becomes the priority because it is what makes more depth safe.

### Rung 1 — keep `--rand-depth`; make it LoopFormer-shaped only if Rung 0 says depth pays

Current sampler is uniform over 1..4. Two evidence-backed variants, in order of preference:
* **(1a) drop T=1 from the sampler** (`T ∈ {2,3,4}`): depth-1 states exist as the loop's first
  iteration inside every depth-4 step, so we lose no calibration and stop spending 25% of steps on a
  readout we never ship. One-line change.
* **(1b) LoopFormer dual trajectory**: every step also runs the full depth-4 trajectory and adds
  `0.1·L_cons` (stop-gradient per-token **logit** consistency from the short trajectory to the long
  one). ~30 lines, ~1.5× step cost. This is the only published recipe that ties short trajectories to
  the deep one, and it is the closest thing to "cheap and works" for elastic depth.
* **Not doing:** a growth curriculum (2609.19107 is 7.4B, data-constrained multi-epoch — wrong regime),
  and shallow-biased sampling (Huang/Huginn argue for it as a *regularizer*; with only 4 depths and a
  compute-matched A/B, uniform is the honest control).

**Collapse-resistance.** There is no learned quantity that scales the loss: the CE is a mean over
executed iterations at the realized depth. Nothing to shrink. The consistency term is a *distance to a
stop-gradient target produced by our own deeper run* — it cannot be gamed by making the target
garbage, because the target's own CE is in the loss at full weight. Failure mode is not collapse but
**waste**: depth 4 degenerating to depth 1. That is directly visible in Rung 0's depth curve (if
`BPB@4 ≈ BPB@1`, kill it) and it is an honest number, not a fake one.

### Rung 2 — inference-only confidence exit on the rand-depth checkpoint (the deliverable the owner actually gets)

Sequence-level (per generated row) first, per-token only if the KV story is solved.

* **Criterion**: softmax **margin** `p₁ − p₂` on the fp32 logits at the prefix mean, per
  2608.18222's `R_t = Σδ_u < μ(z_t)` where available, else a monotone per-depth threshold schedule
  (CALM's decaying-threshold lesson: one threshold at depth ≤ 4 is a cliff).
* **Readout**: exit at k ⇒ use the **prefix mean over 1..k**. Under `--rand-depth` that is the trained
  readout of the depth-k model, so no retraining and no new head.
* **KV**: a sequence-level exit has no missing-KV problem at all. For per-token exit, an early-exited
  predecessor needs a copy-forward policy for its K/V and its KDA state (CALM: copy the last computed
  state, do **not** try to recompute K/V from lower layers).
* **Deliverable**: a measured **frontier**, not a claim: (mean loops, held-out BPB) pairs at 4–5
  thresholds, with the zero-regression point called out. Report BPB in nats/byte, not just the
  percentage of tokens skipped.
* **Why this rung is first among the *shippable* ones**: zero training change, and the measured
  failure mode in the controlled comparison is **overspending** (CALM margin: 5.63 loops, −14 items)
  rather than divergence. RecurTrace's numbers say a confidence rule will *lose a little accuracy* and
  buy a lot of compute. That is a fine trade at depth 4 and it is honest.

**Collapse-resistance.** Nothing is trained. The exit rule is a threshold on a quantity the LM head
already produces.

### Rung 3 — shallow-anchor term (make more depth safe) — only if Rung 0 shows peak-then-fall

Add to the loss, at whatever depth-4 step:

`L_anchor = λ · ‖logits(depth-4 readout) − sg[logits(depth-1 readout)]‖` (stop-gradient on the target),
or the hidden-state version 2608.18222 actually used (pull deep latent states toward the frozen base's
own shallow state, rank-16 LoRA, `r ~ U[16,128]`, 2 000 steps). We already compute the depth-1 readout
inside every depth-4 step, so this is nearly free in FLOPs.

**Collapse-resistance.** The target is our own model's own shallow readout and it is stop-gradient;
the loss is a distance, not a weight on the LM loss. There is no path by which shrinking a learned
quantity lowers the objective. The *real* risk is the opposite one — **capability flattening**:
iterations 2..4 stop adding anything, the loop becomes depth-1, and the depth curve goes flat. That is
a measurable, honest degradation (Rung 0's curve detects it immediately), not a fake loss.
Kill criterion: `BPB@4 − BPB@1` shrinks below the noise floor ⇒ revert. λ is the only knob; sweep
{0.03, 0.1} (the weight scale used by LoopFormer/T-LoopFormer for their consistency terms is 0.1).

### Rung 4 — oracle-distilled decider — only if Rung 0's oracle rate says there is real headroom

Head: `p_t = σ(w^T z_t + b)` on the mean-pooled state, `d ≈ 512` parameters, trained with BCE against
`y_t = 1[min_{t'>t} L_t' < L_t − δ]` computed from the same forward pass (RecurTrace Eq. 6). Hard floor
`T_min = 2`; threshold τ selected on held-out NLL, not on the test set. Sequence-level first (it is
what RecurTrace ships, and it is what their 2.7%-of-items gain looks like); per-token only if the
per-token gather is safe on sm_120 (it is not, today).

**Collapse-resistance, three independent reasons:** (i) the target is a *measured* per-batch quantity,
recomputed every step — the same anchoring that keeps MoR's router scores bimodal at {0,1}; (ii) the
LM loss is **not** weighted by `p_t`, so there is no loss-mass knob (this is the exact difference from
PonderNet's `Σ p_n CE_n`); (iii) the floor makes depth-1 unreachable by construction.
**Kill criteria**: mean depth pinned at the floor for >90% of inputs after warmup (collapse under a
new name); the fake-loss canary fires (unweighted final-state CE vs any p-weighted number, gap ≫ noise
— the channel that produced our frozen-at-uniform eval); and the RecurTrace warning that with
`T_min` set to the best fixed depth the method *"merely reaches fixed-depth parity"*.

### Rung 5 (do not build) — MoR-style expert-choice router

Rejected on stack grounds (per-depth token-subset gather = the sm_120 crash pattern; ≤3 loop
iterations of FLOPs on a launch-bound step), **not** on the 135M claim, which the isoFLOP table shows
is a recursion effect, not a router effect (§2.2). If it is ever revisited, copy T-LoopFormer's router
shape (cumulative-softmax threshold, no depth loss) rather than MoR's (which needs the aux BCE to
patch causality).

### A/B protocol vs fixed-depth-4

All arms: real corpus, `small` preset, batch 10 / s512, same seed, same steps, same held-out window,
sequential (one heavy thing at a time; `--guard --detach --log`; `systemd-run --user --scope
-p MemoryMax=40G`).

| arm | what | primary metric | kill |
|---|---|---|---|
| **A0** | fixed-4 (exists) | held-out BPB @ depth 4 | reference |
| **A0'** | A0 re-evaluated at depths 1,2,3 (Rung 0) | BPB@k, ΔCE, oracle rate, EAS@k | if `BPB@4−BPB@3` ≤ noise → close the question |
| **A1** | `--rand-depth` uniform 1..4 (exists) | BPB@4 vs A0; depth curve 1..4; s/step | BPB@4 worse than A0 by > noise, or s/step +>5% |
| **A2** | `--rand-depth` T ∈ {2,3,4} | same | BPB@4 worse than A1 |
| **A3** | A-winner + `0.1·L_cons` (Rung 1b) | same + prefix-mean vs iteration-k readout gap | BPB@4 worse, or step +>50% |
| **A4** | A-winner + `λ·L_anchor` (Rung 3) | same; **watch `BPB@4 − BPB@1`** | depth-4 flattens onto depth-1 |
| **A5** | A-winner + oracle decider (Rung 4) | same; mean depth; % at floor | floor-pinned >90%, or canary fires |

**Noise floor — be precise about which noise.** The eval window is fixed and rewound, so the *same
checkpoint* scores bit-identically (that was the 2026-09-27 fix). Three distinct floors:
1. **Eval noise (paired).** Same 100 KB, two models: the per-token CE difference is highly correlated,
   so the paired SE is far below the unpaired one. I estimate the resolution at roughly
   **0.002–0.005 BPB**, but that is an estimate, not a measurement — **measure it before the A/B**:
   score one checkpoint at depth 4 and at depth 3 across the 20 batches and take the batch-to-batch
   spread of the difference, or run the same arm twice. Marked **NOT VERIFIED** until measured.
2. **Window bias (not noise).** 100 KB of held-out text is a *systematic* offset from the true corpus
   BPB. It cancels in every paired A/B, so it never matters for a win/lose call — but **absolute BPB
   numbers from this window must never be compared to published BPBs**, and a depth curve measured on
   100 KB can be biased *systematically* if byte difficulty varies along the window. Cross-check the
   depth curve on a second, disjoint window before believing a small depth slope.
3. **Seed noise (the real floor).** Any claim of the form "adaptive depth wins" is a claim about
   training, and the training seed is the dominant term. Two seeds of A0 before claiming any win, and
   a win smaller than the A0 seed spread is not a win. RecurTrace needed 3 training seeds × 8
   non-overlapping eval seeds and an exact McNemar test to defend +2.2 points at 1.7B; a +2.2-point
   effect on 2400 items is a *large* effect by their standards, and our 7.5M expectation is smaller.

**Canaries, every log step, all arms**: held-out BPB; per-iteration ΔCE; depth histogram; the
fake-loss canary (unweighted final-state CE vs any weighted number); the `BPB@4 − BPB@1` "is the loop
still doing anything" gap; NaN count per repo protocol. And one new one from 2608.18222: the
per-step displacement `δ_t` (normalized), so we can tell a *settling* loop from a *drifting* one
without a task suite — settling is the regime where extra depth is free.

---

## 7. What I could not verify

* **NOT VERIFIED:** the paired eval noise floor on our fixed 100 KB window (§6, item 1) — needs a
  measurement, and it gates the "is `BPB@4 − BPB@3` real" call.
* **THE "100 KB window" IN THIS DOCUMENT IS NOT A CONSTANT, and every occurrence
  above is batch-10 shorthand.** The window is `eval_batches × batch × seq_len`; at
  batch 10 × seq 512 it is 102 400 B, at batch 2 it is **20 480 B**, and runs at
  different batch sizes have therefore scored different amounts of text (the
  batch-2 ablations reported `over 20480 B`, the batch-10 recipe
  `over 102400 B`). A BPB is only comparable **within one window**, and the byte
  count printed on the eval line is the authority. Any A/B run at a different
  batch size from its control is void regardless of the numbers.
* **NOT VERIFIED:** whether *any* adaptive-depth mechanism has been measured below 0.6B on natural
  language. The closest small-scale result (2608.18230) is about **which part** of the loop to
  recur, not about per-input depth allocation. There is no source for "adaptive depth pays at 7.5M".
* **NOT VERIFIED:** any behaviour of random-depth training on a byte-level 256-vocab model. Every
  cited looped LM is subword (49K / 32K vocab). Byte difficulty is plausibly *more* skewed
  (space/newline after common words vs rare UTF-8 continuations), which would help TaH-style skipping
  — but that is a hypothesis, and Rung 0 measures it directly.
* **NOT RE-VERIFIED IN THIS SESSION** (inherited from `docs/archive/research/2026-09-26-ponder-replacement.md`,
  not re-opened): DeepLoop 2607.13491, SMELT 2609.01343, Training-Free Looped 2605.23872, Hyperloop
  2604.21254, LoopUS 2605.11011, Mixture-of-Depths 2404.02258, PALBERT 2204.03276, PonderNet
  2107.05407, Arrabal-Campos 2608.22347, Huginn's own Fig. 6 numbers (I verified the caption and the
  training objective, not the plotted values).
* **Cost note:** STARS' Jacobian regularizer is **not implementable** here — it needs JVP, i.e.
  forward-mode AD, and the stack is reverse-mode burn/cubecl. Recorded so nobody re-derives it.

---

## Sources (all opened; arXiv HTML full texts unless noted)

- [R1] Huang, Sun, Liu, Sedra, Weinberger — *Deep Networks with Stochastic Depth*, arXiv:1603.09382v3.
  Eq. 4 (decaying `p_ℓ`), Eq. 5 (test-time re-weighting), §4 tables.
- [R2] Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein — *Scaling
  up Test-Time Compute with Latent Reasoning*, arXiv:2502.05171v2. Eqs. 1–2 (log-normal Poisson,
  r̄=32, σ=0.5), truncated BP k=8, Fig. 6 caption, "Bad Run 1" representation collapse.
- [R3] Jeddi, Ciccone, Taati — *LoopFormer*, arXiv:2602.11451v1. §3.3 loss, Algorithm 1, Tables 1–2.
- [R4] Yu, Zhang, Zhao — *T-LoopFormer*, arXiv:2609.15160v1. §3.2 (Eqs. 2–4), §3.3 loss, Tables 1–4,
  App. A (d=2048, TPOT/memory, "instabilities at depth").
- [R5] Yang et al. — *STARS*, arXiv:2605.26733v1. Abstract, §4.3 (random loop sampling insufficient),
  §5 (JSRR/power iteration/JVP), Fig. 1 + Ouro-1.4B GSM8K numbers.
- [R6] *Retrofitting Recurrent Depth into a Pretrained LM*, arXiv:2608.11233v1. Abstract (extrapolation
  to ~1.5× supervised depth; "Learned depth selection remains open").
- [R7] *How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents*,
  arXiv:2609.19107v2. Abstract + §1 (loop growth vs random loops; compute-optimal loop growth).
- [R8] Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun — *Mixture-of-
  Recursions*, arXiv:2507.10524v3 (NeurIPS 2025). §3.2 + Fig. 3, §4.2 Table 4, §5.2 Fig 5(b), App. G
  Tables 13–14, App. D Table 9. Code: github.com/raymin0223/mixture_of_recursions.
- [R9] Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler — *Confident Adaptive Language
  Modeling*, arXiv:2207.07061v2 (NeurIPS 2022 oral). §3.2–3.5, Fig. 2, App. C. Code:
  github.com/google-research/t5x/tree/main/t5x/contrib/calm.
- [R10] Elhoushi, Shrivastava, Liskovich, Hosmer, Basti, Wasti, Lai, et al. — *LayerSkip*, arXiv:2404.16710v2.
  §4.1–4.3, abstract speedups.
- [R11] *The Diminishing Returns of Early-Exit Decoding in Modern LLMs*, arXiv:2603.23701v1. §3.3
  metrics, §5 findings.
- [R12] Wang, Feng, Shen, Xu, Wang, Wu — *RecurTrace*, arXiv:2609.03379v2. Abstract, Eq. 6, Table 2,
  App. E depth histogram.
- [R13] *Think Shallow, Solve Deep*, arXiv:2608.18222v1. §5.1 (R1–R4, Prop. 1), §5.4, §7.1 (Huginn-3.5B,
  2 585 trajectories, carry probe), App. H (label-free repair).
- [R14] *Allocating Recurrent Compute in Looped Language Models*, arXiv:2608.18230v1. Abstract, §3.1–3.3,
  §4.1 (15.7M / 116.9M, GDN, T=4).
- [R15] *Do Transformers Use their Depth Adaptively?*, arXiv:2604.12426v1. Abstract + §1.
- [R16] Fu, You, Chen, Dai, Yang, Wang — *Think-at-Hard*, arXiv:2511.08577v4. Abstract, §4 (oracle
  12–19% of tokens, +2.0–7.3%), §5.2–5.3 (two-stage decider, depth-aware LoRA at d>1).
- [R17] *Depth-adaptive Inference of Looped LMs via Continuous Depth Batching*, arXiv:2608.09444v1
  (abstract only). *Energy-guided Recursive Model*, arXiv:2607.10128v3 (abstract only). *The Ignition
  Is Real, and It Lives at the Readout*, arXiv:2608.03263v1 (abstract only). *Less is More: Recursive
  Reasoning with Tiny Networks*, arXiv:2510.04871v1 (abstract only). Code for T-LoopFormer:
  github.com/YuMingQian1234/T-LoopFormer; STARS: github.com/njuyxw/STARS; TaH:
  github.com/thu-nics/TaH.
- Internal: `crates/dormouse-core/src/loop_block.rs` (iterations, per-iteration CE, mean readout,
  `set_depth`), `crates/dormouse-core/src/model.rs` (norm→lm_head readout), `crates/dormouse-train/
  src/lib.rs` (`sample_depth`, rand-depth step, the eval block: rewind, `eval_batches=20`, targets=None),
  `docs/adr/0013-fixed-depth.md`, `docs/architecture/PLAN.md`,
  `docs/archive/research/2026-09-26-ponder-replacement.md`.
