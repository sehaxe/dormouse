# One-button pretrain→post-train: what the strongest 2026 pipelines actually do

**Question.** Owner directive: *«одна кнопка: pretrain + post-train идеально; оптимизация и
эффективность важны»*. This document answers: what do published end-to-end pipelines do about
stage management (curriculum, stage boundaries, per-stage sampling), post-train recipe,
verifier and wall-clock — and which of it is evidence-backed for dormouse (byte-level MoE,
1.07B total / ~57M active, ONE 16 GB card, 19.7 GB filtered corpus).

**Method / evidence marks.** arXiv MCP `get_abstract` by ID (no downloads), the repo's own docs
(read-only), **one** HTML method read (Puro-2B). Every ID in the brief was checked individually.

| mark | meaning |
|---|---|
| **(A)** | verified from the paper's own abstract via the arXiv MCP, 2026-10-16 |
| **(M)** | method-level read of the paper's own full text (only Puro-2B, `arxiv.org/html/2608.27370v1`, fetched 2026-10-16) |
| **(L)** | local artifact in this repo (`docs/papers/`, `docs/`), read 2026-10-16 |
| **(S)** | SPECULATION — my inference, no source. Never presented as fact |

`websearch` was **unavailable** (HTTP 403 on three queries, 2026-10-16), so this sweep is
arXiv + local only. **No method read was done for the arXiv papers except Puro-2B** — every
arXiv row is therefore abstract-level evidence, the standard `2026-10-15-agentic-post-training.md`
used. Where a stage boundary is *not* in the abstract I say so instead of filling it in.

---

## TL;DR

1. **The one-button shape that exists at our budget is Puro-2B's: two phases, one decay, a
   curriculum inside phase 2, and checkpoint averaging at the end** — executed on consumer
   RTX 5090 cards, with published token counts, wall-clock and cost **(M)**.
2. **Nobody in 2026 exits a pretrain stage on a loss plateau.** The published exits are
   *token/data-witnessed budgets* (Puro-2B **(M)**, Index-1.9B **(A)**, SmolLM2 **(A)**), a
   *schedule's own decay phase* (WSD family **(A)**), or a *horizon-free schedule that removes
   the decision entirely* (WSqD 2607.10959, Anytime 2602.03702) **(A)**.
3. **Decay is the pipeline's one irreversible act, and 2026 evidence says it may be the wrong
   act before post-training**: Warmup-Stable-Only beats decay-based schedules *after SFT* at
   1B and 8B, because decay lands in sharper minima (2603.16127) **(A)**.
4. **Post-train has converged on one mixed RL run, not an 8-stage ladder**, followed by
   on-policy distillation (repo PLAN verdict **(L)**; Instella-MoE's published order is
   SFT→DPO→RL-with-multi-teacher-OPD, 2609.00791 **(A)**; MiMo-V2.6, 2610.11959 **(A)**).
5. **The only stage-exit signal with a measured early-convergence property** is a mid-scale
   capability probe: induction-circuit identification converges within **0.3–2 % of total
   training tokens** (2606.02378) **(A)** — and Qwen3.8-Flash-Next names evaluation throughput as
   its tightest bottleneck and asks for exactly such a probe **(L)**.
6. **For a 19.7 GB corpus the binding constraint is repetition, not schedule**: beyond a point
   repetition is counterproductive and compute belongs in capacity, and **weight decay λ ≈ 1.0
   cuts the overfit coefficient by ~70 %** (2605.01640) **(A)** — a `run` default, not a project.

---

## 1. The pipeline table

Read as: *what would `dormouse run` have to implement to copy this?* "not reported (A)" means the abstract does not contain the number — not that it does not exist.

### 1.1 Full pipelines (pretrain → post-train)

| pipeline (ID, date) | stage management: curriculum, boundaries, per-stage sampling | post-train recipe + verifier | wall-clock / cost | ev. |
|---|---|---|---|---|
| **Puro-2B** 2B dense, consumer RTX 5090 `2608.27370` (2026-08-27) | **Two phases.** P1 = 438.84B tok, **power schedule** (Shen et al.), 24 GPU; P2 = 959.99B tok, **linear decay** from P1's terminal LR, 96 GPU. **Curriculum = within-source ordering**: examples sorted low→high by *source-provided* scores, chunked, chunks aligned by normalized within-source rank so cross-source mixture weights are **preserved**; no global cross-source quality score. Late stage: **LR held constant for the last 29B tokens, then the last 6 checkpoints are averaged** (curriculum model averaging). Mixture/source selection by **proxy experiments** (small controlled runs → "proxy profile" vector), not human curation **(M)** | Post-train is a **probe, not a product**: matched SFT with identical data/optimization for the two P2 variants. Verifier: none (no RL). Reported result: **curriculum init → higher GSM8K after SFT in repeated runs; broad-instruction 15-task aggregate improves** — pretrain curriculum effects **survive post-training** **(M)** | **17.6 elapsed days total** (P1 10.43 d / P2 7.16 d), 22,514 active GPU-h, **$6.89K** under their rental-equivalent accounting; $4.4K run also reported. MFU ~73 % at 24 GPU (238 TF/GPU), 192 TF/GPU at 96 **(M)** | **M** |
| **Qwen3.8-Flash-Next** (local report, `docs/papers/tech_report-qwen3.8-flash-next.pdf`) | CPT stage split **in steps and tokens, with the reason**: indexer warm-up stage = **1,000 steps × 8 seq × 256K ≈ 2B tok @ LR 1e-3 (indexer only)**, then joint stage = **8,000 steps × 96 seq × 256K ≈ 200B tok @ LR 2.5e-5**. Batch-size warmup **removed**: it cost **18.8 % more optimizer steps for no gain**. Conclusion names "phase-by-phase cost accounting" as the method **(L)** | The report is architecture+optimization; post-training is used as a *validation* instrument. Two lessons it states: **sparse writes that look fine in pretraining degrade after post-training**; and **"the tightest bottleneck is evaluation throughput: a cheaper mid-scale probe that reliably predicts post-training ordering"** **(L)** | 3× fewer tokens and ~1/9 the FLOPs than the 397B-A17B flagship at retained quality **(L)** | **L** |
| **Instella-MoE** 16B/2.8B active `2609.00791` (2026-09-01) | Published order of stages: **pre-training → mid-training → long-context extension → SFT (feedback-driven data curation) → DPO → RL with Multi-Teacher On-Policy Distillation**. Fully open flow: weights, configurations, data mixtures, code **(A)** | Verifier class: multi-teacher OPD + RL. Pre-train benchmark avg **76.7**, post-train Think checkpoint avg **73.2** — the post-train drop is reported, not hidden **(A)** | Trained on AMD MI300X/MI325X; wall-clock not reported (A) | **A** |
| **MiMo-V2.6** (Xiaomi) `2610.11959` (2026-10-08) | **Mid-training on a broad multimodal corpus *before* RL**, explicitly "to provide ample exploration space"; then RL is scaled on three axes — (1) batch/throughput (async, **1,568 samples and 2.7–3.7B tokens per step**, ctx up to 1M), (2) environment diversity, (3) **grader compute**. MoE router **frozen** to keep training stable at scale | Post-train **is** the product: groupwise agentic grading as the reward signal, multi-layer reward-hack defense, honest/reward shaping toward **shorter, token-efficient solutions**. Verifier: executable environments + agentic graders | Wall-clock not reported in the abstract; tokens/step and ctx are (A) | **A** |
| **INTELLECT-3** 106B/12B active `2512.16144` (2025-12-18) | Not a stage ladder: **SFT and RL on top of GLM-4.5-Air-Base**, scaled to **512 H200s**; the contribution is the *stack* (prime-rl, async, first-class multi-turn/tool use) | Verifier = the open `verifiers` library + Environments Hub environment collection; recipe and frameworks open-sourced **(A)** | 512 H200s at "high training efficiency"; elapsed time not reported (A) | **A** |
| **Olmo 3** 7B/32B `2512.13961` (2025-12-15) | Releases "the entire model flow … every stage, checkpoint, data point, and dependency used to build it" — the reference artifact for *stage-level transparency* | Family includes Think (long-CoT) and Instruct variants **(A)** | **Stage token boundaries are NOT in the abstract**; a method read of the HTML is owed if `run` copies their stage list — open question §6 | **A** |
| **SmolLM2** 1.7B `2502.02737` (2025-02-04) | **Multi-stage over ~11T tokens**, mixing web with specialized math/code/instruction data; **mix rates updated at each stage based on the previous stage's performance**; new datasets (FineMath, Stack-Edu, SmolTalk) introduced **where existing ones were too small or low-quality**. Small-scale ablations + manual refinement **(A)** | SFT + DPO on released instruction sets (SmolTalk family) **(A)** | Wall-clock not reported (A) | **A** |
| **Nanbeige4-3B** `2512.06266` (2025-12-06) | **Fine-Grained Warmup-Stable-Decay (FG-WSD)** scheduler: "progressively refines data mixtures across stages" — i.e. the *decay axis and the data axis are one schedule*, not two knobs | SFT with deliberative-generation refinement + CoT reconstruction → **Dual Preference Distillation** from their flagship → **multi-stage RL with verifiable rewards and preference modeling** | 23T tokens, 30M+ instructions; wall-clock not reported (A) | **A** |
| **Index-1.9B** `2607.09885` (2026-07-10) | **WSD** with **curated-data concentration raised substantially during the decay phase** — the decay phase is where the good data goes. Controlled studies on depth, LR magnitude/scheduling, **LR-decay × data-quality interaction**, and instruction data in pretraining. Documents *"an unexplained surge in benchmark performance midway through the constant-learning-rate phase"* | SFT → DPO; plus a retrieval-augmented variant; a "Pure" control with instruction-like data filtered out of pretraining **(A)** | 2.8T tokens; wall-clock not reported (A) | **A** |
| **nanochat** (~124M, Karpathy) — **repo-cited only** | Cited in `docs/architecture/PLAN.md` as "the only complete pretrain→SFT→RL pipeline published at ~124M, three scripts", with **tool use taught inside GSM8K** (8K rows × 4 epochs vs 460K chat rows, ~14:1) rather than as an agentic corpus | RL recipe copied into the repo's plan: no trust region, no KL to a reference, DAPO token-level normalization, advantage = r − μ, 16 samples/example, RL lr = 5 % of SFT lr | Not verified here (**S**: treat every nanochat number as second-hand until re-read from the source) | **L** + **S** |

### 1.2 Post-train-only mechanisms worth a row (they define `run`'s exit criteria)

| mechanism (ID, date) | what it decides | measured result | ev. |
|---|---|---|---|
| **RetireOPD** `2609.20784` (2026-09-17) | **Adaptive Retirement**: the student drops the teacher **on its own** once teacher–student discrepancy stops shrinking **and** it reaches a target fraction of the teacher's success rate; RL continues alone. Also: "privileged information alone does not always make a teacher reliable" and "the benefit of teacher supervision is **stage-dependent**" | Qwen2.5 1.5B→7B: ALFWorld **+14.1…18.8 %** over RL baseline, WebShop **+11.8…19.0 %**, and it **surpasses its own skill-conditioned teacher** | **A** |
| **SEAD** `2606.28562` (2026-06-26) | Where to spend distillation: entropy partitions tokens into zones (≈**50 % skipped**), KL direction **anneals forward→reverse as competence grows**, and a **competence-gated easy→hard** prompt curriculum | On OLMo-3 7B→32B: **+4.8 avg** over vanilla OPD on six math benchmarks; components are **super-additive** (token selection needs the curriculum) | **A** |
| **CRR** `2609.33875` (2026-09-27) | Step-level credit without a learned process RM: fork the env, sample an alternative action, replace the advantage with (real − counterfactual) return | 14B policy, **equal-wall-clock** on the same hardware: SWE-bench Verified **41.7 % vs 36.7 %** outcome-only GRPO; fork overhead is counted, not hidden. Limits: stochastic continuations, costly/imperfect replay | **A** |
| **SEED** `2607.14777` (2026-07-16) | Turns completed on-policy trajectories into hindsight natural-language skills, then converts the skill-induced probability shift into a **dense token-level on-policy distillation signal** | Text + vision agentic tasks: "consistently improves performance and sample efficiency" (abstract truncated mid-sentence in the MCP read) | **A** |
| **Byte-level recipe** `2602.01007` (2026-02-01) | The only published *stage order* for a byte model: **two-stage curriculum** — (1) progressive KD aligning byte representations to a token-teacher's embeddings, (2) byte-level SFT for end-to-end byte generation | Retains most teacher capability with **≈125B bytes** across Llama/Qwen/OLMo teachers — vs "trillions of bytes" from scratch | **A** |

### 1.3 Rejected / set aside, with the reason

- **The 8-stage serial post-train ladder** (Rufus-Air shape, `2609.29421`, repo-cited): the repo
  already ruled it obsolete **(L)** — INTELLECT-3, MiMo-V2.6 and Nemotron 3 replaced it with one
  mixed RL run plus post-RL multi-teacher distillation. Its four *principles* (SFT as capability
  floor, learnability band, order stages by exposure-to-hacking, infra is part of the recipe) are
  kept.
- **Any learned reward model / LLM-judge reward at 57M active**: no 2026 source found where one
  pays for itself at ≤7B on an executable objective; the best measured agentic gain (CRR) needs
  none **(A)**; the judge is itself a documented failure mode (repo doc `2607.07436`).
- **1M-context / sparse-indexer stages**: Qwen3.8's QSA stages are real **(L)** but they are
  *continued-pretraining stages at 256K context*, already parked here (§5 of the readiness plan).
- **SmolLM2's "we invented three datasets" strategy as a template**: it assumes a data budget
  where new corpora can be created. dormouse has 19.7 GB and no data-creation loop; the
  transferable part is only "update mixes from the previous stage's evals" **(A)**.

---

## 2. Stage-boundary decision rules actually published

Every rule below is the *signal a pipeline used to leave stage 1*. Nothing here is a loss plateau — where I found no plateau rule, that absence is the finding.

| # | rule | source | what it looks like in practice | ev. |
|---|---|---|---|---|
| R1 | **Fixed token/phase budget, planned up front** | Puro-2B: P1 438.84B tok, P2 959.99B tok, seq 4096, GBS 1536 **(M)**; SmolLM2 ~11T multi-stage **(A)**; Index-1.9B 2.8T **(A)** | boundaries are budget decisions, not detected events | **M/A** |
| R2 | **The schedule's own decay phase is the boundary** | WSD family (2601.09000 **(A)**); Index-1.9B raises curated-data concentration *in the decay phase* **(A)** | "leaving stage 1" = "starting the cooldown", and the *data mix changes with it* | **A** |
| R3 | **Remove the boundary: horizon-free schedules** | WSqD 2607.10959 (inverse-sqrt base + linear cooldown, one peak LR reused across horizons, minimax-optimal \(O(1/\sqrt T)\) in the stochastic convex model) **(A)**; Anytime Pretraining 2602.03702 (polynomial horizon-free LR + **weight averaging**, 150M/300M to **32× Chinchilla**, comparable to tuned cosine) **(A)** | `run` need not know total tokens; the cooldown is scheduled at exit, not at launch | **A** |
| R4 | **Online, state-driven LR instead of a pre-committed boundary** | SOLAR 2609.34681: bounded, state-dependent residual corrections re-anchored to a base schedule every step, with a **circuit breaker**; 60M→1B dense (AdamW and Muon) + two MoE settings to 3B; matched 130M controls: global PPO controller 27.09 → 23.74 PPL, group-wise 22.87 **(A)** | the boundary becomes a learned, bounded adjustment — the loud failure mode is handled by re-anchoring + circuit breaker | **A** |
| R5 | **Repetition budget: stop when extra epochs stop paying** | 2605.01640: additive overfitting penalty under repetition; "beyond a point, further repetition is counterproductive and compute is better spent on model capacity"; **λ = 1.0 weight decay cuts the coefficient by ≈70 %** **(A)** | *the rule for a 19.7 GB corpus.* Compute `tokens ÷ corpus bytes` (epochs witnessed) as a first-class run metric and let it gate the phase switch | **A** |
| R6 | **Data-quality ratio, not loss: synthetic vs natural** | 2510.01631 (>1000 LLMs, >100k GPU-h): rephrased synthetic alone is **no faster** than natural web; 1/3 rephrased + 2/3 natural speeds up **5–10×** at larger budgets; textbook-style synthetic alone gives **notably higher downstream loss at small budgets**; ratios converge to **~30 %**; generators beyond ~8B params do not help **(A)** | stage sampling policy: cap synthetic ≈30 %, prefer rephrase-style, never bulk pure-generated at small budget | **A** |
| R7 | **Capability probe converges long before the model is finished** | 2606.02378: 3 × 1B-class models, 10 log-spaced revisions each. Induction-circuit formation precedes BOS-attractor formation by **10–20× in tokens** (two transitions, not one); the capability screen converges to the final induction circuit within **0.3–2 % of total training tokens**; OLMo 1B shows a **sharp BOS phase transition (7 %→70 % between adjacent checkpoints)** | the probe is cheap and early — but emergence *shape* differs per model (ramp vs phase transition), so a threshold that works for one architecture may never fire for another | **A** |
| R8 | **A sub-module gets its own warm-start stage, then joint training** | Qwen3.8-Flash-Next: indexer-only **1,000 steps ≈ 2B tok @ LR 1e-3**, then joint **8,000 steps ≈ 200B tok @ LR 2.5e-5** **(L)** | the cleanest published "stage boundary with a reason" in the whole set: boundary = when the new module's own objective has converged | **L** |
| R9 | **"Unexplained" is not a rule** | Index-1.9B documents an unexplained benchmark surge midway through the constant-LR phase, and does *not* give a mechanism **(A)** | recorded here so nobody builds an exit rule on it | **A** |
| — | **Loss plateau: no source found in this sweep** | — | I found **no** 2026 pipeline that exits a pretrain stage on a held-out loss plateau. The repo's own ADR-0013 already rules a *related* learned-halting signal (PonderNet λ collapse) uninformative **(L)** | — |

---

## 3. The most one-button-able pipeline for a ≤1B byte model on ONE card

### 3.1 Verdict

**Copy Puro-2B's structure, not its scale**: *one run, two phases, one terminal decay, a
curriculum inside the second phase, and checkpoint averaging as the only output step* **(M)** —
the only pipeline in this sweep executed on consumer cards with published tokens, wall-clock and
cost, and the only one whose matched-SFT probe shows the pretrain curriculum choice surviving
into post-training **(M)**.

Its four transferable properties for a one-button supervisor:

1. **No stage is entered by detection.** Both phases have token budgets known at launch.
2. **The only irreversible step is the terminal decay + averaging**, and it is deliberately late
   and cheap (last 29B tokens + last 6 checkpoints).
3. **The curriculum preserves mixture weights** by ordering *within* each source and aligning
   sources by normalized rank — no cross-source quality score to fit, which is exactly what a
   one-button tool cannot invent; data selection is **proxy-driven** (small controlled runs →
   per-source "proxy profiles" → mixture weights) **(M)**.

### 3.2 Evidence-backed **at ≤1B total**

| stage shape | who proved it at our size | ev. |
|---|---|---|
| Multi-stage pretrain with per-stage mix updates | SmolLM2 **1.7B** (~11T tokens) **(A)** | A |
| WSD + concentrated curated data in the decay phase | Index-1.9B **1.9B non-embedding** **(A)** | A |
| Two-phase pretrain, Muon-family optimizer, FP8, curriculum + checkpoint averaging | Puro-2B **2B** on consumer GPUs **(M)** | M |
| Complete pretrain→SFT→RL in three scripts at ~124M | nanochat, repo-cited **(L)** | L |
| Two-stage token-teacher→byte-student curriculum (~125B bytes) | 2602.01007, validated on Llama/Qwen/OLMo teachers **(A)** | A |
| Horizon-free / anytime LR (no boundary to detect) | 2602.03702 at **150M/300M**; WSqD on SlimPajama **(A)** | A |
| Adaptive distillation retirement; competence-gated curriculum | RetireOPD **1.5B–7B**; SEAD **7B→32B** **(A)** — *at* our size, not below it | A |

### 3.3 **Unproven at our scale** (do not ship as defaults)

1. **Any byte-level post-training at any scale.** The repo states it plainly: *"zero byte-level
   post-training results exist at any scale"* **(L)**. Every mechanism in §1.2 was measured on
   token-level models ≥1.5B.
2. **Agentic RL at ~57M active params.** Same source: no ≤1B agentic report found in a ~20-query
   sweep; the nearest verified size is Qwen2.5-1.5B (VH-GPO) **(L)**.
3. **A trained critic / PPO value head as the *first* RL mechanism** — the most expensive of three
   credible options at our size; forked-replay advantage and self-distilled credit deliver
   step-level credit without one **(L/A, 2609.33875 / 2609.20784)**. Relatedly, **CMA on a MoE with
   tiny active width is a different experiment than Puro-2B's dense model (S)**.
4. **FP8 blockwise on 16 GB sm_120 at our shapes.** Puro-2B's numbers (val-loss +0.0031…0.0039 vs
   BF16, 98.0 % compute retention, 1.36× throughput → 1.34× net) are RTX 5090 + Megatron/Transformer
   Engine **(M)**; this repo's own fp4/fp8 history is a list of retractions **(L)**.
5. **A `run` that decides its own exit from a loss curve.** No rule in §2 licenses it; the only
   early-converging signal found is a *capability probe* (R7), architecture-dependent in shape.

### 3.4 Recommended stage order (one button)

```
P1  data-witnessed budget, stable/high LR      → checkpoint family (resumable, no decay)
P2  curriculum-ordered data, linear decay,      → constant-LR tail + average last k checkpoints
    mix weights from proxy profiles             → exit rule: repetition/overfit gate (R5), not loss
PT1 SFT floor (byte-level role protocol)        → capability floor
PT2 ONE mixed verifier-RL run (forkable-env)     → step-level credit, no learned RM
PT3 on-policy distillation, ADAPTIVE RETIREMENT → self-terminating stage
PT4 SFT re-interleave (never RL-only)           → anti-collapse
```

Gated, and honest about it: PT2 and beyond are **(L)**-level conclusions from the repo's own
2026-10-15 sweep plus >1.5B agentic papers, **not** a ≤1B byte-level result **(S)**.

---

## 4. What the teams themselves say made it HARD

- **Infra is part of the recipe.** Puro-2B only reaches consumer-card bandwidth with a modified
  driver + IOMMU/ACS/NPS changes (one-way 31.5 → 56 GB/s; 8-GPU AllReduce 14.75 → 27.34 GB/s),
  and GDR via an undisclosed driver patch — **without them the same recipe degrades**. The
  **LM head alone costs compute comparable to several Transformer layers**, fixed by assigning
  fewer layers to its pipeline stage. Cost accounting **excludes failed and exploratory runs** **(M)**.
- **Curriculum ordering and late-stage optimization are one design**: the LR is held constant so
  late high-quality data is not washed out by a decay, then checkpoints are averaged to keep that
  data's influence **(M)**.
- **A wasted optimizer budget is a measurable loss**: batch-size warmup cost **18.8 % more
  optimizer steps for no gain**; its removal is reported as a win **(L, Qwen3.8)**.
- **Reward hacking and router drift are the scale killers**: MiMo-V2.6 freezes the MoE router,
  builds a multi-layer anti-reward-hack defense, and demands train-inference consistency before
  it scales RL **(A)**. Distillation degrades with student competence (incoherent rollouts →
  noisy gradients, mastered tokens → redundant ones; ≈50 % of tokens skippable) and needs the
  curriculum to work at all **(A, SEAD)**; a privileged teacher is unreliable and stage-dependent,
  so supervision must be retired on evidence **(A, RetireOPD)**.
- **Replay/fork cost is the hidden tax**: CRR counts fork overhead explicitly and still bounds
  itself to environments with affordable, reliable state restoration **(A)**. Data law bites
  first: repetition overfits **(A, 2605.01640)** and pure-generated textbook data raises
  downstream loss at small budgets **(A, 2510.01631)**.
- **The measurement layer is the bottleneck, not the training**: Qwen3.8's conclusion names
  evaluation throughput as its tightest bottleneck and asks for a cheaper mid-scale probe that
  predicts post-training ordering **(L)** — the same lesson this repo already paid for twice
  (an eval that scored a different network; a window that varied with batch size) **(L)**.
- **Seed variance can exceed the effect at small scale**: at 151M, seed variance was larger than
  every recipe difference measured, so ≥3 seeds per arm are required **(L)** — which is why a
  one-button run must log seeds and windows, not only loss.

---

## 5. Actionable design decisions for `dormouse run`

Each row: what to build, the source that licenses it, and the A/B that makes it ours.
| # | decision for `run` | licensed by | confidence / A/B pointer |
|---|---|---|---|
| **D1** | **Make the whole pretrain a single resumable two-phase run whose ONLY irreversible act is a terminal decay + checkpoint-average step.** Launch policy = horizon-free base LR (inverse-sqrt / polynomial) + weight averaging, so `run` never needs to be told the total token budget; the decay is scheduled at *exit*, from the pre-decay checkpoint family. | Puro-2B **(M)**; WSqD 2607.10959 **(A)**; Anytime 2602.03702 **(A)** | High for the structure; the specific LR *shape* is ours → A/B: inverse-sqrt base vs WSD-constant at the same tokens, 3 seeds, one batch size, `docs/protocols/AB-PROTOCOL.md` |
| **D2** | **Do not decay before post-training by default.** Tie "decay" to "this checkpoint is the release artifact", and let the SFT/RL branch start from a **stable-LR** checkpoint: WSO beats decay-based schedules *after SFT* at 1B and 8B, including mid-training and over-training regimes. | 2603.16127 **(A)** (decay → sharper minima; WSO → flatter) | Medium-high (1B/8B token models, not byte-level) → A/B: decay-vs-stable at the SFT handoff, judged after matched SFT, not on pretrain BPB |
| **D3** | **Gate the P1→P2 switch on data-witnessed tokens and a repetition-overfit proxy, and set data-constrained weight decay high (λ ≈ 1.0).** Publish `epochs_witnessed = tokens / corpus_bytes` on every log line. | 2605.01640 **(A)** — repetition becomes counterproductive; λ=1.0 cuts the overfit coefficient ~70 % | High for the rule, ours for the constant → A/B on the 100M rung: 1 vs 2 vs 4 epochs × λ ∈ {0.1, 1.0}, BPB + a downstream probe |
| **D4** | **Stage sampling policy = cap synthetic at ~30 %, rephrase-style, never bulk pure-generated at small budget; order data within-source and align sources by normalized rank (no cross-source quality score).** | 2510.01631 **(A)**; Puro-2B **(M)** | Medium (synthetic study is token-level and large-budget) → A/B against the current `docs/research/2026-10-15-data-strategy.md` mix |
| **D5** | **Post-train = SFT floor → ONE mixed verifier-RL run → self-retiring on-policy distillation → SFT re-interleave; verifier-first, no learned RM in v1.** The distill stage must end on its own signal: discrepancy stopped shrinking AND student ≥ target fraction of teacher success. | Instella-MoE's published order and MiMo-V2.6's one-scaled-RL-run **(A)**; RetireOPD 2609.20784 **(A)**; SEAD 2606.28562 **(A)**; SFT/RL interleave and rollback **(L)** | Medium at our size (mechanisms measured ≥1.5B) → A/B: interleaved vs RL-only, and teacher-retirement on/off, both with the MI/entropy collapse monitor |
| **D6** | **Build the *probe* before the long run: a mid-scale capability probe + frozen eval window is a first-class `run` output, and every stage decision must be readable off it.** Capability circuits converge within 0.3–2 % of total tokens, and evaluation throughput is the named bottleneck of a production pipeline. | 2606.02378 **(A)**; Qwen3.8-Flash-Next conclusion **(L)**; repo eval-window/byte-count rules **(L)** | High → no A/B needed; the check is that the probe fires early and predicts the final ordering (report both, never one) |

Sequencing: **D6 before D1** — it is the instrument every other decision is read from, and this
repo's history is a list of decisions taken on a broken instrument **(L)**. **D5's RL half is the
least proven piece at our scale** and should be wired last.

---

## 6. Open questions / owed work

1. **Olmo 3's stage token boundaries are NOT verified here** (abstract only). If `run` copies an
   explicit stage table, read `2512.13961` at method level first.
2. **Every arXiv row is abstract-level**; `SEED`'s abstract was also truncated in the MCP read, so
   no claim beyond the repo's 2026-10-15 doc is made for it. A method read of `2605.01640` (the
   repetition law D3 rests on) and `2603.16127` (D2) is the cheapest way to raise confidence on
   the two highest-leverage decisions.
3. **Wall-clock is missing from almost every source** — Puro-2B is the only row with real elapsed
   days. Any dormouse wall-clock estimate is **(S)** until measured on this box.
4. **`docs-site/tools/manifest.mjs` needs a line for this document** and `python3 tools/check_doc_refs.py` must be green — **owed, not done**: this task was scoped to exactly one file (nothing committed).
5. **`websearch` was 403 for every query** on 2026-10-16; a 2026-10 pipeline outside arXiv would
   not have been seen by this sweep.

## Sources (all fetched 2026-10-16)

arXiv MCP `get_abstract`: `2602.01007` · `2608.03392` · `2609.33875` · `2610.11959` · `2607.14777` ·
`2609.20784` · `2510.01631` · `2502.02737` · `2605.01640` · `2512.06266` · `2609.00791` ·
`2607.09885` · `2603.16127` · `2609.34681` · `2607.10959` · `2602.03702` · `2608.27370` ·
`2512.16144` · `2512.13961` · `2606.28562` · `2606.02378` · `2609.30063`.
arXiv MCP `search_papers`: 15 relevance-sorted queries (cs.CL/cs.LG/cs.AI, 2025-06-01 → 2026-10-16)
across mid-training curriculum, data curriculum, LR decay schedules, byte-level LMs, small-model
technical reports, OLMo, Nemotron, INTELLECT, end-to-end recipes.
Method-level: `arxiv.org/html/2608.27370v1` (Puro-2B). Local: `docs/papers/tech_report-qwen3.8-flash-next.pdf`
(`pdftotext -layout`, lines 404–500 and 1650–1700) · `docs/architecture/PLAN.md` ·
`docs/architecture/post-training.md` · `docs/research/2026-10-15-agentic-post-training.md` ·
`docs/decisions/2026-10-15-1b-readiness-plan.md` · `AGENTS.md` · `port/README.md`.
