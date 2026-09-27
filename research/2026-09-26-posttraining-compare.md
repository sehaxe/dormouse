# Post-training recipes compared: is Rufus-Air the one, and what transfers to a 7.5M byte model?

Date: 2026-09-26 · Question: owner asked whether Rufus-Air (arXiv:2609.29421, 8-stage post-training on GLM-4.5-Air-Base 106B-A12B) is THE BEST post-training recipe, or whether something better exists. Dormouse is 7.5M params, byte-level, growing — so the real question is **which recipe philosophy generalizes to small models**, not which recipe wins at 106B.

All sources opened and read: arXiv abstracts + Rufus-Air full HTML, Tülu 3 full HTML, INTELLECT-3 technical report (PDF→text), MiMo GitHub + HF model card, SmolLM3 blog, SmolTulu (arXiv API), ufakzeka-1 (arXiv API). Links inline.

---

## 1. The six recipes side by side

### 1.1 Rufus-Air (Amazon, 2026-09-24) — [arXiv:2609.29421](https://arxiv.org/abs/2609.29421)

- **Base / scale**: GLM-4.5-Air-Base, 106B total / 12B active MoE. SFT on 512×H200; RL stages on 8–32 nodes.
- **Stages (serial, 8)**: SFT → Reasoning RL → Coding RL → Instruction-Following RL → General Agent → Coding Agent → Search Agent → RLHF.
- **Rewards**: Reasoning/Coding RL = deterministic verifiers (Math-Verify, generated Python checkers, fuzzy match), binary, "low-noise, programmatically auditable". IF RL = rubric LLM judge. Agent stages = rule-based assertions + execution tests + LLM-judged trajectories. RLHF = preference reward model, last.
- **SFT data**: 9.01M samples, 44.5B raw tokens, 27B loss tokens (6 categories, all from 17 public datasets, zero in-house regeneration). **Key finding: eval scores flatten within epoch 1 while loss keeps dropping — they ship the mid-plateau checkpoint (3799), not the last one.**
- **Core novelty**: not any single algorithm but (i) SFT as a *capability floor* (SFT-only already beats the official GLM-4.5-Air release on IFEval +24, AIME +6.6/+3.5), (ii) *learnability filtering* everywhere (pre-training: drop prompts solved >80% or never solved; online: keep groups with mean reward in (0, 0.8]; DAPO-style dynamic sampling in agent stages), (iii) **stage order = exposure to reward hacking**: hard verifiable rewards first, gameable judge rewards last; the one exception (IF RL early despite a rubric judge) is explained — instruction following is close to what the policy already does, so the judge has little room to be gamed, (iv) infrastructure as part of the recipe (token-in/token-out on-policy rollouts, one chat template from SFT through agents, stateful sandboxes, Rollout Routing Replay).
- **Honesty section (§6)**: stage order is "one that worked, not one shown to be optimal"; Coding Agent stopped at compute budget, not convergence; no single benchmark tracked across all stages; unexplored alternative = multi-teacher on-policy distillation.
- **Code**: no dedicated repo. Recipe = the paper; stack is open-source components (Slime, Megatron, SGLang); data all public.
- **Result**: leads the official GLM-4.5-Air release on every reported benchmark except Arena-Hard v2 Creative Writing; biggest stage deltas are Multi-challenge +24.7 (IF RL) and Tau2-Retail +9.8 (General Agent).

### 1.2 Tülu 3 (AI2, 2024-11, v5 2025-04) — [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)

- **Base / scale**: Llama-3.1 8B / 70B / 405B (dense). Smallest proven: **8B**.
- **Stages (3)**: SFT (~939K-sample public mix) → DPO (on-policy preference data) → **RLVR last** (GSM8K + MATH + IFEval constraint checks, PPO with rule-based rewards).
- **Core novelty**: coined/popularized *RLVR* (RL with Verifiable Rewards); the full-open discipline — every dataset, the training code ([open-instruct](https://github.com/allenai/open-instruct)), decontamination tooling, and a dev/unseen eval split (IFEval-OOD, HREF) so you can see whether your recipe generalizes or just overfits its own dev benchmarks. Also publishes "Insights from the Unfruitful" (online DPO, rejection sampling didn't reliably help).
- **Data findings that held**: scaling unique prompts > reusing SFT prompts for DPO; on-policy preference data (model's own completions as chosen/rejected base) beats off-policy.
- **Code**: fully open (code + data + eval).

### 1.3 INTELLECT-3 (Prime Intellect, 2025-11-26) — [blog](https://www.primeintellect.ai/blog/intellect-3), [tech report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf)

- **Base / scale**: GLM-4.5-Air base, 106B MoE. 512×H200, two months.
- **Stages (2+1)**: general chat-and-reasoning SFT (~207B tokens: OpenReasoning-Math/Code/Science/Tool + AM-DeepSeek-R1-Distill; Muon optimizer) → smaller agentic SFT (tool-call formats, long context to 98K) → **one large mixed RL stage** (Math, Code, Science, Logic, Deep Research, SWE environments all together; 256 prompts × 16 rollouts, 65K ctx, async off-policy, Muon).
- **Core novelty**: infrastructure-first — async-only trainer (prime-rl), [verifiers](https://github.com/PrimeIntellect-ai/verifiers) + Environments Hub (500+ community RL environments as pinnable modules), Prime Sandboxes (20K+ prebuilt images, sub-10 s startup at massive concurrency). Algorithm: masked token-level importance sampling (IcePop-style double-sided masking, α=0.5/β=5) after GSPO/CISPO-collapse ablations. Difficulty pools (easy/normal/hard by Qwen3-4B solve rate), online filter drops pass-rate-1 prompts.
- **Honesty**: rewards and benchmarks still rising, no plateau — the released checkpoint is a budget checkpoint, not a converged one.
- **Code**: all open (prime-rl, verifiers, sandboxes, weights).

### 1.4 MiMo-V2.6 (Xiaomi, ~2026-09-22) — [HF card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), [open RL dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)

- **Base / scale**: own base. Pro-RL 1T, Flash-RL 309B-A15B MoE, omni, 1M ctx. Frontier scale.
- **Stages**: cold start (incl. "aligned RL: cold start from self-correction") → **one mixed RL run** — "You Only RL Once": coding, general agents, visual, cybersecurity tasks from multiple harnesses in the same batch (the opposite of Rufus-Air's serial per-domain stages) → **MOPD2** multi-teacher on-policy distillation to extend capabilities to hard-to-verify tasks.
- **Core novelty**: *scaling the reward signal itself* — Groupwise Agentic Grading: GRS (Groupwise Reward Synthesis: build task-specific rubrics offline from contrasting rollouts) + GAR (Groupwise Advantage Redistribution: rank passing trajectories online, move advantage to higher-quality ones, prefer shorter paths). Binary pass/fail can't rank passing solutions, so the grader is the scaled resource. Async GRPO at 1,568 prompts × 16 rollouts; environment hardening + adversarial screening + verifier cross-checks "keep the loop honest against reward hacking".
- **Code**: weights + RL dataset open; technical report PDF on HF; no training-code repo.

### 1.5 Open-Reasoner-Zero (StepFun/Tsinghua, 2025-03) — [arXiv:2503.24290](https://arxiv.org/abs/2503.24290)

- **Base / scale**: Qwen2.5-32B **base** (no SFT at all).
- **Stages**: one — pure RL from base.
- **Core novelty**: minimalism proven: vanilla PPO + GAE(λ=1, γ=1) + rule-based reward + **no KL regularization** scales both benchmark performance and response length, matching DeepSeek-R1-Zero at 1/10 the training steps. Critic learns to devalue repetitive outputs. Every extra trick (KL penalty, curated SFT floor) is shown unnecessary for the reasoning-RL phenomenon itself.
- **Code**: open (code, data, weights).

### 1.6 SimpleRL-Zoo (HKUST, COLM 2025) — [arXiv:2503.18892](https://arxiv.org/abs/2503.18892)

- **Base / scale**: **10 base models, Qwen2.5 0.5B → 32B**, Llama3-8B, Mistral-7B/24B. The only one of the six with sub-1B evidence.
- **Stages**: one — zero RL (rule-based math rewards) from base.
- **Core novelty**: the *negative results*: zero RL is not plug-and-play across model families. Qwen bases already self-reflect before RL (so Qwen-based reproductions overstate how general the phenomenon is); increased response length does not imply the "aha moment"; the aha moment appears in non-Qwen small models only with careful **format-reward adjustment and query difficulty control**. Design strategies, not a fixed recipe, transfer.
- **Code**: open (code, models, analysis tools).

---

## 2. What's common — the load-bearing invariants

| Invariant | Rufus-Air | Tülu 3 | INTELLECT-3 | MiMo-V2.6 | ORZ | SimpleRL-Zoo |
|---|---|---|---|---|---|---|
| Verifiable/rule-based rewards before judge- or model-based ones | §2.1: "reward reliability sets the stage order"; verifiable stages first, RLHF last | RLVR as the final capability stage; preference data on-policy from the model itself | all RL env rewards rule-based/verifiable | "verifier cross-checks keep the loop honest against reward hacking" | "straightforward rule-based rewards" | rule-based rewards only |
| Difficulty filtering (keep the productive band) | drop solved >80% and never-solved; online keep mean reward ∈ (0, 0.8] | (earliest form: RLVR data limited to verifiable domains) | pools easy/normal/hard; online filter drops pass-rate-1 | "each problem undergoes careful cleaning and difficulty assessment" (MiMo-7B); V2.6: GRS/GAR grade within groups | length/repetition handled by critic | "controlling query difficulty" is one of the two key designs |
| SFT floor quality (RL refines, doesn't build) | §3.1: "SFT is a capability-building stage, not a warm-up"; evals flatten in epoch 1 | ~939K-sample curated SFT mix is the base of the whole pipeline | two SFT phases "establish a strong prior and stable behavioral foundation" | cold-start SFT before RL (aligned RL from self-correction) | **explicit counterexample**: RL works directly on base | **counterexample too**: zero RL works from base — but only for reasoning-shaped tasks and only with difficulty control |
| One consistent output format end-to-end | token-in/token-out rollouts, "chat template handling kept consistent from SFT through the agentic stages" | chat-template consistency studied as a variable (App. B.3) | chat template section; tool-call format standardized across all datasets | harness mixing requires format discipline | n/a (single-turn boxed answers) | format reward is a *tuned* knob, base-model dependent |
| Publish what didn't work | §6 "On what the recipe does not establish" | §8.2 "Insights from the Unfruitful" | reports collapse ablations (GSPO vs CISPO) | aligned RL + anti-hacking sections are failure-driven | shows the minimalist baseline is enough | the whole paper is largely negative results |

The syntheses also agree with each other across time: Rufus-Air explicitly cites Tülu 3 as the exception to thin reporting; INTELLECT-3's difficulty machinery is Rufus-Air's learnability filter with different vocabulary; MiMo-V2.6's anti-hacking language is Rufus-Air's ordering principle made operational inside a single run.

**The interesting 2026 disagreement**: serial per-domain stages (Rufus-Air: 8 stages, each fixes a capability) vs. one mixed run (INTELLECT-3: everything in one RL stage; MiMo-V2.6: "You Only RL Once", tasks from multiple harnesses in the same batch "so capabilities reinforce each other and strategies transfer to harnesses never seen in training"). The field is moving from serial to mixed, with distillation after RL for the hard-to-verify remainder (MiMo's MOPD2; Rufus-Air names multi-teacher on-policy distillation as its unexplored alternative). Serial stages are the safer, more debuggable form; mixed runs are the more compute-efficient frontier form. Nobody has shown serial > mixed or vice versa at matched compute.

---

## 3. What's proven at small scale (<1B, and the 1–8B band)

**Sub-1B evidence is thin but real, and all of it is RLVR/zero-RL or SFT-only:**

- **SimpleRL-Zoo** includes **Qwen2.5-0.5B** — zero RL with rule-based rewards produces real reasoning gains at 0.5B, but only with tuned format reward + query difficulty control. This is the *only* sub-1B RLVR data point in the six recipes.
- **SmolTulu** ([arXiv:2412.08347](https://arxiv.org/abs/2412.08347)): the Tülu 3 pipeline (SFT → DPO, no RLVR) adapted to SmolLM2-1.7B → SOTA sub-2B IFEval (67.7, +11) and GSM8K (51.6). Methodologically important: they ran the recipe ablations on a **135M model** and found the LR/batch ratio effect is large and *task-dependent* (reasoning tasks want higher LR/batch, pattern-recognition tasks lower). So recipe hyperparameters do NOT transfer unchanged down-scale even when the stage structure does.
- **ufakzeka-1** ([arXiv:2609.25081](https://arxiv.org/abs/2609.25081), 2026-09, 151M Turkish model, ~$286): post-training = SFT on open+generated data, no RL. Two findings directly relevant to dormouse: **(a) training-seed variance was as large as the spread across every recipe tried — single-seed comparisons at this scale are uninformative; (b) some skills (identity tracking, multi-turn arithmetic) did not move under any data change — a capacity ceiling, not a data gap.**
- **SmolLM3-3B** ([blog](https://huggingface.co/blog/smollm3)): the fully-open small-model recipe — reasoning **mid-training** (35B tokens of R1 traces) → SFT (1.8B tokens, 4 epochs, mix of think/no-think) → **APO** (Anchored Preference Optimization, an off-policy DPO variant) → model merge with a mid-training checkpoint to recover long-context. **No RL at all**, deliberately: "most existing approaches involve complex reinforcement learning processes and proprietary datasets". At 3B they chose off-policy distillation-style alignment over on-policy RL.
- **MiMo-7B** ([arXiv:2505.07608](https://arxiv.org/abs/2505.07608), the small-scale ancestor of V2.6): RL on a 7B with **130K rule-verified math/code problems**, only rule-based accuracy rewards "to avoid potential reward hacking", test-difficulty-driven fine-grained code rewards (dense signal on hard problems), easy-problem re-sampling for stability. Their headline thesis: *"the effectiveness of RL-trained reasoning relies on the inherent reasoning potential of the base model"* — they rebuilt pretraining to make RL work at 7B. They release Base / SFT / RL-Zero / RL checkpoints, i.e. the SFT-vs-RL-Zero-vs-SFT→RL ablation, open.

**Scale ladder of the six main recipes**: SimpleRL-Zoo 0.5B–32B · Tülu 3 8B–405B · ORZ 32B · INTELLECT-3 106B · Rufus-Air 106B · MiMo-V2.6 311B–1T. **Nobody has published a post-training ablation at 100M–7.5M for a byte-level model.** The arXiv record for "byte-level + instruction tuning" contains only ByT5-style task fine-tuning ([2405.10625](https://arxiv.org/abs/2405.10625)) and hobby-scale national-model reports — no byte-level post-training *recipe* exists anywhere. Everything below is extrapolation from token-level small-model evidence.

---

## 4. For dormouse: what transfers, and the minimal ladder

### What transfers at all

The stage *logic* is tokenizer-agnostic. Rule-based verifiers (exact match, regex constraints, unit tests in a sandbox) operate on **decoded text**; a byte model can learn to emit `\boxed{...}`, diffs, and tool-call formats as byte sequences just as a BPE model does. Nothing in RLVR requires a 150K vocab. What does NOT transfer cheaply: long-CoT reasoning RL (a 7.5M model cannot carry the long deliberation traces that drive RLVR gains at ≥0.5B — the "reasoning potential of the base" precondition of MiMo is exactly what dormouse is still building), and agentic RL infra (Rufus-Air §6: sandbox fleet alone ≈ $10K/month; INTELLECT-3: 44 of 60 nodes for inference rollout).

### The minimal ladder (cheap → expensive, stop where gains stop)

1. **SFT floor (Tülu 3 / Rufus-Air doctrine).** Small but *diverse* byte-level mix: chat, code, tool-call trajectories, math with boxed answers — one consistent format, loss-masked user/tool turns. Scale anchors: Tülu 8B ~939K samples; MiMo-7B cold start ~500K instances (later scaled to 6M); SmolLM3 1.8B tokens. For 7.5M params: start at **10K–100K high-quality examples** (~10–100M bytes), 2–4 epochs, and ship the checkpoint where *held-out evals* flatten, not where loss stops (Rufus-Air's epoch-1 finding). This stage buys format, coverage, and parseable answers — the precondition for every verifier later.
2. **Verifiable RLVR on formats + short tasks (Tülu 3 RLVR + SimpleRL-Zoo + MiMo-7B).** Rewards: exact-match on decoded bytes (math answers), IF-constraint regexes (IFEval-style, the RLVR-IFeval dataset logic), unit tests for tiny code tasks (MiMo's test-difficulty-weighted code reward is the right shape for coding focus). **Difficulty filter is mandatory at this scale**: pre-filter to tasks with observed pass rate in (0, ~0.8] (SimpleRL-Zoo + Rufus-Air band; DAPO-style online filtering later if rollout budget allows). Prompt pool: even 1K–10K verified tasks is in the right order of magnitude relative to MiMo-7B's 130K at 1000× the parameters.
3. **DPO/APO or small-scale distill instead of on-policy RL (SmolLM3/SmolTulu branch)** — if step 2 stalls: generate chosen/rejected from dormouse's own samples (on-policy DPO data beat off-policy in Tülu 3's ablation) or from a larger teacher; off-policy alignment is what the 1.7–3B open recipes actually converged on below ~4B.
4. **Skip until the model is ≥100M and step 2 is clearly working**: judge/rubric rewards (IF-RL-style), agent-stage RL (sandbox environments), RLHF. Rufus-Air's ordering principle says these come last *because they are the most gameable*; for dormouse they are also the most expensive and least likely to produce signal at 7.5M.

### What to copy from whom

- **From Tülu 3**: the discipline — decontamination, dev/unseen eval split, and publishing negative results. For dormouse: hold out a byte-level eval set before building any post-training data, and log every failed recipe variant.
- **From Rufus-Air**: (a) SFT is the capability floor — don't expect RL to fix a bad SFT; (b) learnability filtering as an automatic curriculum; (c) order stages by reward reliability; (d) fixed format/template across all stages. Also its humility: stage order was never ablated — don't treat the 8-stage shape as sacred.
- **From MiMo-7B (not V2.6)**: the proof that RL pays at small scale *only after the base is pretrained for the skill* — dormouse's lever is pretraining corpus/code density first, post-training second. And the dense-reward trick (score test cases by difficulty) for coding RL.
- **From INTELLECT-3**: don't build RL environments from scratch — verifiers/Environments Hub reward functions are text-level and adaptable to byte-decoded outputs.
- **From SimpleRL-Zoo**: expect base-model-specific quirks; format reward is a tuning knob, not a constant; response length growth ≠ capability growth.
- **From MiMo-V2.6 / MOPD2**: the direction post-training is heading (mixed RL + on-policy distillation after RL) matches the distill/self-evolve stages already in dormouse's POST_TRAINING.md — nothing there needs revision, the frontier just validated the order.

### Honest uncertainty

1. **Zero byte-level post-training results exist.** The entire ladder is an extrapolation from BPE models. The first dormouse-specific risk is format: verifiers that assume whitespace-invariant token text may mis-parse byte streams; the second is that byte-level sequences are ~4× longer per content token, so any long-CoT-style RL is even less viable than the parameter count alone suggests.
2. **7.5M is ~60× below the smallest published RLVR success** (Qwen2.5-0.5B). ufakzeka-1's finding — at 151M, **seed variance exceeded every recipe difference they measured** — means dormouse A/Bs need ≥3 seeds per arm or they measure noise. Budget for that before concluding "recipe X doesn't help".
3. **Capacity ceiling is real**: ufakzeka-1 found some skills do not move at 151M under any data change. If multi-step tool use doesn't emerge post-training at 7.5M, that's the expected capacity outcome, not a recipe failure — the growth plan is the recipe that matters.
4. **The serial-vs-mixed question is unresolved** (Rufus-Air 8-stage serial vs INTELLECT-3/MiMo single mixed run) and irrelevant at dormouse's scale: with compute for only one RL arm, run the single most verifiable task family first (math exact-match or unit tests), which both philosophies agree on.
5. **ORZ's counterexample stands**: at ≥32B, reasoning RL works with no SFT and no KL. Nobody has shown SFT floors matter below ~1B; they matter there in every small-model report, but it is possible the floor doctrine is a "big model re-learning formats" artifact — worth a cheap A/B (SFT→RLVR vs RLVR-from-base) once step 2 exists.

---

## Sources

- Rufus-Air — arXiv:2609.29421 (abstract + full HTML, incl. §2.1 ordering rationale, §3.1 SFT, §6 limits)
- Tülu 3 — arXiv:2411.15124v5 (abstract + full HTML: stage table, RLVR section, "Insights from the Unfruitful")
- INTELLECT-3 — [Prime Intellect blog](https://www.primeintellect.ai/blog/intellect-3) + [tech report PDF](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) (SFT tables, RL recipe, difficulty pools, GSPO/CISPO ablation)
- MiMo-V2.6 — [MiMo-V2.6-Flash-RL model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), [XiaomiMiMo HF org](https://huggingface.co/XiaomiMiMo) (V2.6 collection, MiMo-V2.6-RL-oss dataset), [MiMo GitHub](https://github.com/XiaomiMiMo/MiMo)
- MiMo-7B — arXiv:2505.07608 + GitHub README (RL data/reward design, base/SFT/RL-Zero/RL ablation)
- Open-Reasoner-Zero — arXiv:2503.24290
- SimpleRL-Zoo — arXiv:2503.18892 (COLM 2025)
- SmolTulu — arXiv:2412.08347
- SmolLM3 — [HF blog](https://huggingface.co/blog/smollm3) (mid-training/SFT/APO/merge recipe)
- ufakzeka-1 — arXiv:2609.25081 (151M post-training + seed-variance finding)
- Byte-level gap — arXiv API query "byte-level" + "instruction tuning" (3 results, none a post-training recipe)
