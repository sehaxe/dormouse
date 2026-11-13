# Replacing PonderNet halting in dormouse's looped block — research report (2026-09-26)

**Question.** Our 7.5M byte-level LM with a shared-weight loop (max_iter=4) and PonderNet halting (p_n = λ_n·Π(1−λ_j), loss = Σ p_n·CE_n + β·KL) collapses: λ→0 early, p_dist → [9e-4, 4e-6, 1e-8, 2e-11], rec ≈ 0 (fake loss — garbage predictions weighted by ~0), eval frozen at uniform. KL-direction fix didn't help: collapse saves rec ≈ 5.5 while costing β·KL ≈ 0.03. Fixed-depth trains fine. **What replaces PonderNet halting, with no collapse mode, simple in Rust/burn, KV-efficient?**

**Method.** Primary sources only (arXiv full texts incl. v3 HTML, official repos); 6+ rounds; cross-checks between independent papers flagged inline. All numbers below are quoted from opened sources; gaps are marked "no direct source".

---

## 1. Mixture-of-Recursions (MoR) — the deep dive [R1]

Bae et al. (KAIST/Mila/Google), arXiv:2507.10524v3, NeurIPS 2025. 135M–1.7B Llama-style models, N_r recursions of a shared Middle-Cycle block, routers assign per-token depth. Code: github.com/raymin0223/mixture_of_recursions.

**Router designs (§2.2.1).**
- **Expert-choice**: at each recursion step r, scalar score g_t^r = G(θ_r^⊤ H_t^r) (G = sigmoid or tanh); top-k tokens by score pass through the block, rest keep H_t^r unchanged (Eq. 2.1, a residual gate g·f(H)+H). Hierarchical filtering: only tokens selected at step r are re-evaluated at r+1 (simulates early exit). Capacity is a **β-percentile threshold over the batch's scores**; for N_r=3 the paper sequences top-k fractions as 3/3, 2/3, 1/3 to match a perfectly balanced token-choice budget.
- **Token-choice**: one routing decision at step 1, softmax/sigmoid over N_r "experts", top-1; token then runs exactly i blocks (Eq. 2.2). No leakage (causal), but load balancing needs help.

**Router losses (§4.2, Tables 4/13/14, App. G).** This is the load-bearing detail:
- Expert-choice top-k is **non-causal** (a token's selection depends on later tokens' scores), so two causality patches exist: an *auxiliary router* (separate small net, grad-blocked, BCE against the main router's top-k targets) or an *auxiliary loss* — BCE on the main router itself, targets = its own top-k membership (selected→1, unselected→0). **Aux loss wins** for inference-time behavior.
- Final recipe: **aux loss coeff 0.001, sigmoid, plain Linear router, output scaling α=0.1** → NLL 2.8667, few-shot 40.1, dead-token ratio 0.1%, top-k prediction accuracy 99.2% (Table 13). z-loss optional (negligible to NLL; slightly fewer dead tokens).
- Failure corners they hit: aux-router with tanh/α=1.0 → **66.7% dead tokens** (positional bias); some un-normalized configs → **NaN NLL** (Table 13 rows 1–2, 7). Token-choice final recipe: softmax + MLP + **balancing loss (Switch-style α·Σ f_i·P_i) coeff 0.01 + z-loss 1e-3** → MaxVio 0.682 (Table 14); balancing loss beat loss-free bias updates on MaxVio.

**Why routing does not collapse the way PonderNet's λ does.** Four structural differences, each tied to our measured failure:
1. **Selection is rank-based, not scale-based.** Exactly k tokens pass each step *no matter what the absolute score values are*. If all scores shrink, the top-k still fills; the aux BCE then pushes the selected back toward 1. There is no global mass knob the optimizer can profitably shrink to zero. PonderNet's p_n *is* that knob: it multiplies the loss, so shrinking it shrinks the loss — which is literally our measured failure (collapse saves rec 5.5, costs 0.03).
2. **The router output is anchored to a per-batch recomputed target** (own top-k membership), i.e., a self-supervised classifier with BCE — not a preference over loss mass. Result: scores bimodal at {0,1} with "perfect separation" (§5.2, Fig 5b).
3. **Every token traverses at least the first recursion** (§2.2.2) — no depth-0 mode exists, so the model cannot route all tokens to "no compute".
4. **The LM loss is computed once per token at its final depth.** There is no p-weighted mixture of per-depth losses to game. Independent 2026 evidence that the mixture is the fake-gain channel: Arrabal-Campos et al. show that "PonderNet-style halting returns a halting-weighted mixture of hidden states… equalizing the readout annihilates the apparent advantage of native execution (residual +0.000 [0.000, 0.000])" [R3c] — the same mechanism as our rec≈0 fake loss.

**Scale caveat (Fig 3, §3.2).** At the smallest size, 135M, MoR **underperforms the vanilla transformer** ("likely due to a recursive capacity bottleneck"); the gap closes at >360M. Our neighbor is 3.5M tokens of context above dormouse's 7.5M and it's already the regime where adaptive recursion stops paying. No direct source studies MoR-style routing at ≤10M params.

**Stability notes.** MoR reports no λ-analog collapse anywhere; the documented instabilities are router-local (dead tokens, MaxVio, the NaN rows above), all fixed by the aux/balancing losses — because those losses anchor the router to a recomputed target (point 1–2 above).

## 2. Mixture-of-Depths — the parent idea [R2]

Raposo et al. (DeepMind), arXiv:2404.02258. Per-block router r = w^⊤x; top-k (per sequence, per block) above the β-percentile get r·f(x)+x, the rest pass through the residual (Eq. 1). Capacity C is user-set; static graph.

- **Expert-choice chosen over token-choice** for three reasons (§3.3): no aux balancing loss needed; relative weights let the router *compete* for compute ("routers can ensure the most critical tokens are among the top-k by setting their weight appropriately — not possible with token-choice"); two paths → one top-k splits tokens cleanly.
- **Router weights sit on the gradient path** (only for processed tokens) — that is the whole training signal; **no balancing loss is used at all** (§3.4). The aux machinery from MoR appears here only as the *causality patch* for sampling: (a) BCE on the router (selected vs not), costs ~0.2–0.3% on the LM objective; or (b) a grad-blocked MLP predictor, costs ~0 and reaches **99% top-k prediction accuracy** quickly (§3.5, Fig 6: "minimal performance degradation" switching to predictor-based routing).
- **Failure modes documented**: stochastic routing (top-k over Gaussian-random weights) is "drastically worse" than both MoD and baseline (Fig 3) — routing must be learned; token-choice drops surplus tokens arbitrarily. Best config: route every *other* block, capacity **12.5%** (§4.1).
- Note the direction reversal vs PonderNet: MoD's pressure is to *skip* compute for easy tokens while capacity guarantees the hard tokens get it. There is no term that rewards shrinking total compute — capacity fixes it a priori. That is the anti-collapse trick in one sentence.

## 3. PonderNet's documented failure modes and shipped fixes [R3]

- **Original spec** [R3a]: loss = L_NLL + β·KL(p_n ‖ p_G(λ_p)), truncated-geometric prior; **β = 0.01 in every experiment** (parity, bAbI 10k, PAI; Tables 3/5), N=20, λ_p tuned as "inverse expected number of ponder steps". The paper itself contains no collapse analysis — no direct source in the original for λ→0.
- **PALBERT (NeurIPS 2022)** [R3b]: PonderNet's sampling-based exit "introduces major variance in exit layer indices, significantly reducing the resulting model's performance"; fix = deterministic Q-exit (expected halting instead of sampling) + architectural revisit. A shipped fix for exit instability, not for collapse.
- **RecurTrace (2026)** [R3d] — the strongest published replication of *our exact failure*, in the looped-LM setting: on the same backbone, **"ACT and PonderNet collapse to one loop"** (mean depth 1.00, Table 2), and sweeping each method's regularization over **{0.001, 0.01, 0.1} leaves all six settings at mean depth 1.0** — "the collapse is not tied to one hyperparameter choice". Mechanism given: both penalties (expected-depth / geometric prior) "reward stopping early… this generic pressure dominates the oracle quality signal". Their fix, and the caveat that matters for us: with a **hard two-loop floor**, PonderNet+floor reaches *exactly* fixed-depth-2 parity (54.71%, Table 2) — i.e., the floor neutralizes the collapse but the halting machinery then buys **nothing over fixed depth**. They also state the collapse is "a property of attaching penalty-based halting post hoc to a frozen backbone, not a refutation in the end-to-end setting" — but our measurement shows end-to-end-from-scratch on bytes collapses too, so both regimes fail in practice.
- **Fixes shipped in the literature**: (i) KL-direction/beta tuning — measured insufficient here and at all β sweeps in [R3d]; (ii) deterministic exit [R3b]; (iii) **hard depth floor** [R3d] — the only one that reliably removes collapse; (iv) oracle/loss-improvement supervision replacing priors entirely [R3d]; (v) entropy bonuses or minimum-halt floors inside PonderNet itself — **no direct source found** (searched arXiv full-text "PonderNet": only 5 hits total, none proposes entropy bonuses).
- **Instrumentation critique** [R3c]: halting-weighted mixture readouts make adaptive-depth gains unverifiable; equalizing the readout (single-state readout, forced depth) erased the advantage. Directly relevant to how we audit any replacement (see A/B protocol, fake-loss canary).

## 4. The "train fixed, exit early" line (CALM / LayerSkip) [R5]

- **CALM** (Schuster et al., NeurIPS 2022 oral) [R5a]: train the model normally; at *inference*, each token exits when a confidence measure (softmax top-prob, margin, etc.) crosses a threshold, with sequence-level constraints and a fix for attending to missing hidden states of early-exited predecessors. Up to **3× speedup** while "provably maintaining high performance". No training-time halting signal exists → **no collapse mode by construction**.
- **LayerSkip** (Elhoushi et al., ACL 2024) [R5b]: to make early exits *accurate*, training adds layer dropout (rate increasing with depth) + an early-exit loss where **all layers share the same LM head**; inference exits early or runs self-speculative decoding (early draft, full-model verify): **2.16×** on CNN/DM summarization, 1.82× coding, 2.0× TOPv2.
- **Think-at-Hard** (Fu et al., ICML 2026) [R6g] quantifies the headroom this exploits for looped models: a "latent overthinking phenomenon — most token predictions are already correct after the first pass, but are sometimes revised into errors in later iterations"; an oracle iteration policy gains up to 7.3%; TaH skips iterations on **93% of tokens** while beating always-iterate by 3.8–4.4%. (Subword-scale, 0.6B+ models — see byte-level caveat in §7.)
- **RecurTrace's independent comparison** [R3d]: CALM (top-prob) reaches 53.04% at 4.15 loops, CALM (margin) 54.12% at 5.63 loops vs best-fixed 54.71% — i.e., confidence exit **overspends** compute but never collapses; its failure mode is mild inefficiency, not divergence. Contrast with PonderNet/ACT (collapse to 1.0 loops).

**For a 7.5M byte model**: this line is the cheapest to adopt — training stays exactly as it is (our fixed-depth loop already works), and the exit knob lives only in `generate`/`serve`. Bytes are a regime where per-token difficulty is extreme (space/newline after common words vs rare UTF-8 continuations), so the 93%-skippable tail [R6g] is plausibly *larger* at byte level — but **no direct source** measures iteration statistics for byte-level 256-vocab looped models.

## 5. 2026 looped-transformer SOTA scan — what replaced PonderNet? [R6]

Seven 2025–2026 works checked (abstracts + full texts where noted). **Answer: trained halting has vanished from the frontier. Every one ships fixed depth, and none uses PonderNet.**

| Paper | Loop design | Adaptive depth? |
|---|---|---|
| **DeepLoop** (2607.13491) [R6a] | Post-LN DeepNorm looped; tied-depth residual scaling α=(2N)^{1/2}, β=(8N)^{-1/2}; visit-alignment coefficient κ_R formalizes why shared-weight depth needs different residual scaling | **No** — fixed loop count; "improves validation loss once recurrent depth is activated" |
| **SMELT** (2609.01343) [R6b] | MoE, middle half of layers **looped twice**, budgets matched (FLOPs, params, **KV cache**) up to 54B; Chinchilla-style scaling laws; saves **6.8–18.0%** training FLOPs on the compute-optimal frontier | **No** — fixed ×2 |
| **Training-Free Looped** (2605.23872) [R6c] | Retrofits looping onto *frozen* checkpoints as damped Euler sub-steps (ODE view); +2.64pp MMLU-Pro on Qwen3-4B | **No** — fixed wrapper loop |
| **Hyperloop** (2604.21254) [R6d] | begin/middle/end blocks, middle looped, hyper-connections on the residual stream; ~50% fewer params at depth-matched quality | **No** |
| **Recurrent-depth "Huginn"** (Geiping 2502.05171) [R6e] | 3.5B params / 800B tokens; loop count **sampled randomly (clamped log-normal Poisson) during training**, unrolled to arbitrary (fixed) depth at test time; no halting head at all | **No halting head** — depth is a test-time knob, trained by random-depth exposure |
| **LoopUS** (2605.11011) [R6f] | Post-training recast into encoder/looped-block/decoder; selective gate + random deep supervision + **confidence head** for early exit | Confidence-rule exit (train-time halting absent); RecurTrace: LoopUS-Conf "overspends" at 3.2 loops [R3d] |
| **TaH** (2511.08577) [R6g] | Lightweight **decider** triggers latent iteration only for tokens likely wrong after pass 1; duo-causal attention across iterations; depth-aware LoRA | Supervised per-token decider (mismatch oracle), not preference halting |

Cross-check: RecurTrace [R3d] independently reaches the same conclusion from the adaptive side — the only halting that beat fixed depth in their controlled comparison is **supervised by an oracle** (loss-improvement), never by a prior/penalty, and even then only with a hard floor. Geiping's random-depth training [R6e] is also the *training* half of RecurTrace's recipe — the two newest looped-LM works converge on **fixed/random depth for training + supervised or confidence signals for inference**, with PonderNet/ACT consigned to baseline status (the methods that collapse).

Also noted for our loop: DeepLoop's [R6a] core claim — *tied-depth residual scaling must account for parameter revisits* (shared update aggregates gradients from repeated visits) — is a standing caveat for our ReZero-style loop residual; our learned per-iteration scalars are the low-rank version of their fix, and our fixed-depth arm trains fine, so no action.

## 6. KV-efficiency: recursion-wise caching vs recursive KV sharing [R1 §2.2.2, §4.3, App. H]

MoR's two variants, with measured costs (their Table 2, normalized to vanilla):

| | Recursion-wise caching | Recursive KV sharing |
|---|---|---|
| Mechanism | Cache only the tokens *selected* at each depth; attention restricted to them | Cache all KV **once at the first recursion**, reuse at every depth |
| KV memory | (N_r+1)/2N_r of vanilla | **1/N_r** of vanilla |
| Attention FLOPs | (k/N_ctx)² of vanilla | k/N_ctx of vanilla |
| Complexity | Per-depth gather/scatter of token subsets (dynamic shapes) | None (skip KV projection + prefill at shared depths, Cycle only) |
| Quality | Better for expert-choice | Degrades expert-choice, **helps** token-choice |

Why sharing works at all: decomposing KV per depth shows depths of a *shared* block have "highly consistent magnitude patterns and high cosine similarity" (App. H), so from-scratch pretrained recursive models tolerate sharing with only slight loss (§4.3).

**For dormouse (N=4, s512, byte-level):** recursive KV sharing is the simpler and KV-optimal choice (1/4 memory, zero gather machinery), and our hybrid attention (KDA recurrent state + MSA top-k) has an even stronger sharing story: the KDA state *is* the recursive state. But honesty first: at s512 the KV cache is small and not a bottleneck — KV choice only becomes decisive if we adopt **per-token early exit at generation** (then exited tokens have missing deeper KV, and sharing is exactly MoR's fix for the "missing KV problem" [R1 §2.1, R5a]). Recursion-wise caching would additionally require dynamic token-subset gathering on 4D tensors — the exact pattern that crashes cubecl on sm_120 (repo doctrine, AGENTS.md). Verdict: sharing, and only if an exit mechanism is implemented.

---

## 7. Verdict for dormouse

Ranked. Each item tied to (a) our measurements and (b) citations above.

### Rank 1 — Delete PonderNet halting; keep the loop at fixed depth. *(Do now.)*
- **Why**: our only measured-good arm is fixed-depth; every collapse-replication paper converges on the same conclusion, and the entire 2026 looped SOTA (SMELT, Hyperloop, DeepLoop, Training-Free, Huginn) ships fixed depth — trained probabilistic halting is absent from all of it [R6]. At our scale specifically, adaptive recursion is contraindicated: MoR *underperforms vanilla at 135M* [R1], 18× our size; RecurTrace's healthiest halting variant merely *equals* fixed depth [R3d]. Our failure (collapse saves rec 5.5, costs 0.03) is the mixture-readout fake-loss channel that [R3c] independently shows makes PonderNet-style gains unverifiable.
- **Design**: remove halt head, λ/p_dist, KL term, out_acc from `loop_block` + `loss`; L_Rec accumulation stays as-is (already per-iteration, never materializes [N,b,t,d] — keep). Keep max_iter=4. New checkpoint name (config change → ADR-0005 hard-error on resume is correct behavior here).
- **Rust/burn cost**: net deletion, ~50–150 lines removed, <20 added. Lowest risk of anything on this list. **No collapse mode exists** — there is no learned halting signal left to collapse.

### Rank 2 — Add random-depth training on top of Rank 1 *(the cheap adaptive-depth option; one checkpoint serves depths 1–4).*
- **Why**: the only 2026 mechanism that gives per-input depth flexibility **without any halting preference to collapse**: sample the loop count during training (Huginn: clamped log-normal Poisson over 0–~50; mean ≈3.9 in RecurTrace's reuse) and compute the loss at the sampled depth [R6e, R3d]. Because our per-iteration CE is already accumulated inside `forward_full_state`, the diff is: sample T, truncate the accumulation, backprop through T steps. Loss is full CE at the realized depth — **no p-weighted mixture exists, hence no fake-loss channel and no collapse mode** (the structural property that PonderNet lacks, §1 point 4).
- **Design**: per-batch T ~ Uniform{1..4} (start lazy; upgrade to clamped log-normal only if U is degenerate). Inference exposes T as a knob (BPB-vs-depth curve = the test-time scaling measurement for free).
- **Rust/burn cost**: ~20–40 lines in `loop_block`/`train_loop`. Per-batch (not per-token) sampling → no gather, no dynamic shapes, sm_120-safe.

### Rank 3 — Inference-only confidence exit (CALM-lite) on the fixed model *(generate-time FLOP savings, zero training change).*
- **Why**: train fixed, exit early has published speedups (CALM ≤3× [R5a]; LayerSkip ≤2.16× [R5b]) and its failure mode is overspending, not collapse (CALM: 5.6 loops but 54.1% — never diverges [R3d]). TaH shows most subword tokens need ≤1 iteration at their scale [R6g]; byte level plausibly more so, unmeasured (no direct source — our A1/A2 arms measure it).
- **Design**: in the `generate` loop, after iteration k compute max-softmax-prob of the fp32 logits (we already cast to fp32 there); break when > τ. τ calibrated on held-out for ≤0.01 BPB regression at the operating point. If per-token exit is wanted in batched contexts, KV semantics = MoR's recursive sharing (§6) — reuse iteration-1 KV at later iterations; at s512 this is a non-issue.
- **Rust/burn cost**: ~20 lines in `serve`/`generate` + a one-off calibration pass.

### Rank 4 — Oracle-supervised halt head *(only if Rank 2's measured depth-benefit is real).*
- **Why**: the only trained-halting design with 2026 evidence of not collapsing: a small head predicts p_t = σ(w^⊤ pool(h) + b), trained by **BCE against y_t = 𝟙[min_{t'>t} L_{t'} < L_t − δ]** — whether another loop actually reduces loss, computed from the same forward pass; inference loops until p_t < τ, with a **hard floor T_min** and no penalty/prior term [R3d]. No collapse mode because (i) labels are anchored to measured loss improvement, recomputed every batch (same anchoring that keeps MoR routers bimodal, §1); (ii) the LM loss stays full CE at the final depth (no mixture to game); (iii) the floor makes depth-1-only unreachable by construction — RecurTrace: "the floor guarantees the depth while the head allocates it" [R3d]. For us the oracle is nearly free: per-iteration CE already exists in the loop.
- **Design**: head = 1×d Linear on mean-pooled state (d=512-ish → ~500 params); δ from held-out per-iteration ΔCE scale (start δ = 0.05·CE_1); T_min=2; BCE weight 1.0 (RecurTrace's BCE coefficient: **no direct source** in the sections read — treat as a tunable, sweep {0.1, 1}); log mean depth + share-at-floor every log step.
- **Rust/burn cost**: ~80–120 lines. Only worth it if Rank 2 shows held-out BPB improving meaningfully from depth 3→4 on real data; otherwise it's machinery that returns fixed-depth parity ([R3d] Table 2: even PonderNet+floor = fixed T=2 exactly).

### Rejected — MoR expert-choice router at 7.5M.
Contraindicated four ways: (a) scale — MoR loses to vanilla at 135M [R1]; (b) machinery — non-causal top-k forces aux BCE + capacity sequencing + hierarchical filtering + one of two KV strategies [R1, R2]; (c) our stack — per-depth token gathering is dynamic token-subset indexing, the pattern that crashes cubecl on sm_120 (repo doctrine); (d) payoff shape — the KV/FLOP wins MoR banks at 2K context don't exist at s512, and our training steps are launch-bound (repo perf notes: batch6/s512 ≈ 9.7 s/step A/B-identical after a CE-path change), so router-saved FLOPs may not even buy wall-clock. Same logic demotes token-choice (needs balancing loss, MaxVio never reached 0 even after long training [R1]) and TaH's duo-causal attention (a new attention axis — an architecture rewrite for a 7.5M model).

### A/B protocol (sequential arms; repo GPU/RAM doctrine — one heavy thing at a time)
All arms: real corpus carve, `small` preset, batch10/s512, same seed, same steps, `--eval` held-out tail.
1. **A0 fixed-4** (exists) — reference.
2. **A1 fixed-{1,2,3}** — measures whether iterations 3–4 pay at all. *If held-out BPB is flat across 2→4, the entire adaptive-depth question is closed for this model class: answer = "iterations don't help bytes here," delete Rank 2–4 permanently.*
3. **A2 random-depth {1..4}** (Rank 2) — report BPB at each forced depth (one ckpt) + s/step. **Kill**: BPB@4 worse than A0 by > noise band, or s/step +>5%.
4. **A3 oracle halt head** (Rank 4) — only if A2 shows a real 3→4 depth slope. **Kill**: mean depth pinned at floor for >90% of tokens after warmup (= collapse recurrence under a new name), or the fake-loss canary fires.
- **Canaries logged every log-step, all arms**: held-out BPB; per-iteration ΔCE; for any p-weighted signal: **fake-loss detector** = CE(unweighted final state) vs Σp·CE_n — gap ≫ noise ⇒ reject the arm outright (this is the exact channel that produced our frozen-at-uniform eval [ours; R3c]); halt/depth distribution drift over steps; NaN watch per repo protocol.

### Bottom line
Nothing should *replace* PonderNet at training time — the 2026 frontier's answer is that trained preference-halting is the bug, and it gets deleted: **fixed depth now (Rank 1), random-depth training as the cheap universal checkpoint (Rank 2), confidence exit as the inference knob (Rank 3), oracle-supervised head with a hard floor only if depth provably pays (Rank 4)**. Every one of these has no loss-weighted mixture in the training objective, which is precisely the channel our collapse exploits.

## Sources

- [R1] Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun — *Mixture-of-Recursions*, arXiv:2507.10524v3 (NeurIPS 2025). Full text: arxiv.org/html/2507.10524v3; code: github.com/raymin0223/mixture_of_recursions.
- [R2] Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro — *Mixture-of-Depths*, arXiv:2404.02258. Full text: arxiv.org/html/2404.02258v1.
- [R3a] Banino, Balaguer, Blundell — *PonderNet: Learning to Ponder*, arXiv:2107.05407v2 (β=0.01, KL(p_n‖p_G(λ_p)); full HTML text inspected).
- [R3b] Balagansky, Gavrilov — *PALBERT: Teaching ALBERT to Ponder*, arXiv:2204.03276 (NeurIPS 2022).
- [R3c] Arrabal-Campos, Montoya, Alcayde, Fernández — *Where Cognition Lives…*, arXiv:2608.22347 (mixture-readout equalization nulls halting gains).
- [R3d] Wang, Feng, Shen, Xu, Wang, Wu — *RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory*, arXiv:2609.03379v2. Full text: arxiv.org/html/2609.03379v2 (Table 2; §5.2 collapse analysis; §3 oracle Eq. 6).
- [R5a] Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler — *Confident Adaptive Language Modeling*, arXiv:2207.07061 (NeurIPS 2022 oral).
- [R5b] Elhoushi et al. — *LayerSkip*, arXiv:2404.16710 (ACL 2024).
- [R6a] Li, Zhang, Guo, Gu, Wang — *DeepLoop*, arXiv:2607.13491v2. [R6b] Wang, Zhang, Luo, Wu, Liu, Liu, Huang, Yan, Li — *SMELT*, arXiv:2609.01343v3. [R6c] Chen, Li, Liang, Lao, Liu — *Training-Free Looped Transformers*, arXiv:2605.23872. [R6d] Zeitoun, Torroba-Hennigen, Kim — *Hyperloop Transformers*, arXiv:2604.21254v3. [R6e] Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein — *Scaling up Test-Time Compute with Latent Reasoning*, arXiv:2502.05171v2. [R6f] Park, Lee, Kim, Bae — *LoopUS*, arXiv:2605.11011. [R6g] Fu, You, Chen, Dai, Yang, Wang — *Think-at-Hard*, arXiv:2511.08577 (ICML 2026).
- Internal: repo `AGENTS.md` (sm_120 no-4D-dynamic-slice; launch-bound step times; resume/config doctrine); our measured PonderNet failure (session context, 2026-09).
