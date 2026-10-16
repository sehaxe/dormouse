# Data mixing for pretraining, 2024-2026 — methods, shipped recipes, and a mix plan for `b1` (2026-10-16)

Scope: domain-level data mixing for pretraining (and briefly SFT), and what **we** should do with one
19.7 GB filtered English-dominant byte corpus and a 5-20B byte budget. Read-only pass; **only this file
was created**, nothing committed, no GPU work.

**Source note.** The arXiv MCP server rate-limited this IP (HTTP 429 on every batch and on single serial
retries), so ids were resolved by fetching arxiv.org abs/html, ACL Anthology and ICLR-proceedings pages
directly. Every number was read on a resolved primary page.

## TL;DR

1. **The mixing *search* got ~550× cheaper by 2026; the *gain* did not grow.** FastMix (ICLR 2026) 1.3
   GPU-h vs RegMix; DeMix (ICML 2026) needs no per-mixture training; OptiMer (Mar 2026) 15-35× cheaper.
   But the EACL-2026 survey says most methods' optimal weights are "only slightly better than uniform
   sampling, or even worse", and MEDU finds **token-count heuristics beat manual and learned mixes**
   (2501.11747). For one GPU: don't buy a law.
2. **The lever worth paying for is *which* data sits in the decay phase.** WSD's loss drop lives in
   cooldown (2508.01483, 2410.05192), and high-quality annealing data beats plain WSD by **+3.5% MATH /
   +2.9% HumanEval / +5.5% MMLU-Pro** (2507.17634, ICLR 2026). SmolLM2's stage-4 upsample is where math
   goes 3.21→22.07 and code 8.87→23.21 (2502.02737).
3. **Code is cheap general-purpose insurance at our scale** (+8.2% rel. NL reasoning, +4.2% world
   knowledge, 12× code vs text-only, 470M-2.8B, ICLR 2025). Shipped mixes sit at **13-17% code**. But
   "more is always better" is **false**: HumanEval pass@20 is non-monotone, 22.6 → 15.9 (2610.11548).
4. **At ≤1 epoch over 19.7 GB repetition is not our constraint** — but code/math/RU will repeat, and the
   2026 rule is per-domain: the optimal repeat count tracks the **domain's own loss** and rises mildly
   with model size at fixed tokens-per-parameter (2608.14071).
5. **Synthetic: no bulk for us** (~30% rephrased is optimal, but the 5-10× needs *larger* budgets and
   textbook-alone **hurts** at small budgets, 2510.01631).
6. **Concretely:** a 3-stage static mix over the existing corpus + ~1.5 GB of acquired code/math/RU + a
   coarse 6-cell sweep at 9.2M. Not a fitted law.

## 1. Methods and laws (what each optimizes, cost, measured gain)

| method | what it does | cost | measured gain | source |
|---|---|---|---|---|
| **DoReMi** | Group-DRO over domains on a small proxy → static weights; task-agnostic | 280M proxy ×2 runs = **8%** of the 8B target's compute | **+6.5 pts** avg one-shot downstream; baseline accuracy **2.6× faster** (75k vs 200k steps), Pile | 2305.10429 (NeurIPS 2023) |
| **RegMix** | many tiny models on random mixtures → regression → train the predicted mix | 512 × 1M-param/1B-token proxies ≈ **10% of DoReMi** | predicted mix best of **64** candidate 1B/25B models; single-task variation ≤ **14.6%**; beats human selection; matches/exceeds DoReMi | 2407.01492 (ICLR 2025) |
| **Data Mixing Laws** | parametric loss-vs-mixture function + nested step/size laws | small-scale fitting runs | 1B/100B RedPajama mix ≈ default mix trained for **48% more steps** | 2403.16952 (ICLR 2025) |
| **BiMix** | bivariate law: domain proportion **and** data volume jointly | fitting runs; entropy proxies **training-free** | loss extrapolation **MRE < 0.2%**, unseen mixtures **R² > 0.97** | 2405.14908 |
| **ODM** (online) | multi-armed bandit re-defines proportions **every step** | negligible wall-clock (reward = information gain) | next-best final ppl with **19% fewer iterations**; **+1.9%** rel. 5-shot MMLU; 1B/50B | 2312.02406 |
| **CLIMB** | embed+cluster into 20 groups, iterative proxy search + predictor | proxy runs; releases ClimbLab 1.2T / ClimbMix 400B | 350M & 1B on 40B beat DoReMi and RegMix; 1B/400B **+2.0% over Llama-3.2-1B** | 2504.13161 (NeurIPS 2025) |
| **FastMix** (2026) | bilevel: ratios ≡ per-source loss weights → gradient descent, **one** proxy | **1.3 GPU-h** pretrain, 2.2 GPU-h SFT | avg **48.2** pre-train, **×550 vs RegMix, ×55 vs CLIMB**; SFT **65.4, +5.5** over next best | ICLR 2026 (Tan et al.) |
| **DeMix** (2026) | one model per domain, **merge weights** → unlimited proxies, no extra training | no training per candidate mixture | Spearman **ρ = 0.81** rank fidelity vs real runs; 1.7B/50B; releases 22T corpora | 2602.00747 (ICML 2026) |
| **OptiMer** (2026) | post-hoc: one CPT model per dataset, merge distribution vectors, Bayesian search | **15-35× lower** search cost | beats data-mixture and model-averaging baselines; Gemma 3 27B, JA/ZH/Math/Code | 2603.28858 |
| **MEDU / UtiliMax** | token-size heuristics + LLM-estimated utility, risk-adjusted | MEDU **~200× cheaper** than ablations | **token-count heuristics outperform manual and learned mixes**; UtiliMax up to 10.6× over manual | 2501.11747 |

**2026 state of the field.** Two surveys agree on the open problems: a static (rule-based / learning-based)
vs dynamic (adaptive / externally-guided) taxonomy, and **limited transferability across domains,
objectives, models and validation sets + unstandardized evaluation** as the cross-cutting failures
(2604.16380; `2026.findings-eacl.15`). **Is mixing still worth it?** VERIFIED: the search is now cheap and
2026 moves the problem post hoc; VERIFIED: the payoff is contested (the two sources above). **DERIVED:**
for a single-corpus, single-GPU run a fitted law is not worth it — a static stage recipe + our own 9.2M
sweep is. (Other 2026 work, untabulated: Data Mixing Agent, ACL 2026, avg 46.3%, +1.45 over RegMix; MixMin
2502.10510; response-surface design of proxy mixtures 2608.23922.)

## 2. Static recipes that actually shipped

| model | mix | source |
|---|---|---|
| **Llama 3** (15T tok) | **~50% general knowledge / 25% math+reasoning / 17% code / 8% multilingual** | 2407.21783 |
| **Nemotron-4 15B** (8T tok) | **70% EN natural language / 15% multilingual NL / 15% code** (53 natural, 43 programming langs) | 2402.16819 |
| **SmolLM2** (1.7B, ~11T tok, 4 stages) | S1: 60% FineWeb-Edu / 40% DCLM + 10% StarCoderData · S2: 20% code + 5% OWM · S3: web 40/60 + Stack-Edu + ~10% math · S4 (**decay, 10% of steps**): 58% web + 4% Cosmopedia + 24% Stack-Edu + 14% math. Eval deltas by stage: math **3.21 → 3.7 → 7.27 → 22.07**, code **8.87 → 10.56 → 16.75 → 23.21** | 2502.02737 (Table 3) |
| **Qwen2.5** (18T tok) | 7T→18T "with focus on knowledge, coding, and mathematics", mixture set by Qwen2-Instruct classifiers; **no fractions published** | 2412.15115 |
| **Nemotron-3 Super** (25T tok, 2026) | 16 categories; **code ≈ 14%**; WSD; only 8-10T tokens open | docs.nvidia.com/nemotron |
| **SwallowCode / SwallowMath** (50B ablation) | 82.2% text / 13.0% code / 4.79% math (code run 84/16); rewritten code **+17.0 HumanEval / +16.1 HumanEval+**, rewritten math **+12.4 GSM8K / +7.6 MATH** | 2505.02881 |
| **FineWeb-Edu vs DCLM** | SmolLM2 runs 60/40 then 40/60; FWE wins MMLU/ARC/OpenBookQA, DCLM wins HellaSwag/CommonsenseQA | 2502.02737 Table 1 |

Shipped code share is **13-17%**, math 5-25% and concentrated late; the shared shape is quality-web early
→ code → code/math decay. Qwen2.5 is the one family publishing no fractions.

## 3. Domain upsampling rules

**Code — beneficial, not monotone.** +8.2% rel. NL reasoning, +4.2% world knowledge, +6.6% generative
win-rate, 12× code vs text-only at 470M-2.8B; a cooldown containing code beats no cooldown by 52.3%
win-rate and cooldown-without-code by +4.1% (ICLR 2025, "To Code or Not To Code?"). Code raises cross-task
**variance** and helps arithmetic "up to a point" (2409.04556), and is **non-monotone and mixture-context
dependent**: HumanEval pass@20 peaks at **22.6** when the target-aligned source is 75% of a 30% controlled
slice, dropping to **15.9** — below the zero-synthetic baseline — at pure curated data (2610.11548, Oct
2026). "More code always better" is **refuted**; **SPECULATION:** no code-fraction→HumanEval curve exists
at 1B/5-20B bytes.

**Math.** The decay upsample is where it pays (SmolLM2 3.21→22.07, 2502.02737); the best small-budget source in that sweep is FineMath4+/3+ (10B/34B tok); rewriting/filtering beats adding volume (+12.4 GSM8K / +7.6 MATH, 2505.02881); synthetic textbook math **alone hurts** at small budgets (2510.01631).

**Multilingual (and Russian).** The curse is scale-dependent: real at ≤45M params, where high-resource languages consistently lose (2311.09205); at **1.1B and 3B across 25-400 languages** there is no significant curse, EN works as a pivot, and balance is what matters (2510.25947). ATLAS gives a 38×38 cross-lingual transfer matrix and the scratch-vs-finetune crossover (2510.22037); code in multilingual CPT helps classification, slightly hurts generation (2504.04152). **Our RU position, honestly: we have no RU eval carve** (`AGENTS.md` §2.6 — one English-dominant 500 MB tail), so any RU claim is unmeasurable today. **SPECULATION:** EN:RU ≈ 95:5 as a decay upsample.

**Repeat / epochs.** ≤4 epochs ≈ free (2305.16264), but that "4" is data-size dependent — E(K,N) ≈ K for small K, saturating at **Θ(log N)**, so larger datasets can be repeated more (2511.13421). The 2026 domain-level rule: at fixed tokens-per-parameter the optimal repetition **mildly increases with model size**, is **strongly negatively correlated with the domain's own final loss**, and only weakly related to the amount of unique domain data; Math's minimum was ~5 repeats while Code/Wiki/Medical peaked earlier; a proxy at the same TPP estimates it (2608.14071). **DERIVED:** global epochs are irrelevant for us; cap code/math/RU at ≤4 and decide per domain from the domain's loss — which needs the carve.

## 4. Synthetic data (2026 state)

Rephrased synthetic **alone** is not faster than natural web; **1/3 rephrased + 2/3 natural** gives
**5-10×** but only at larger data budgets; the "good" ratio converges to **~30%** (HQ flat ~30%, QA falls
50%→30% as scale grows); textbook-style alone is **worse** than natural, especially at small budgets;
generators **>8B params are no better** than ~8B; no collapse for rephrased data, collapse-like patterns
for textbook mixtures (all 2510.01631, EMNLP 2025). 2026: megadoc rephrasing plateaus near **1.48× data
efficiency** at 32 rephrases/doc, tuning epoching on synthetic beats capping it at one epoch, and real
data must stay in the mix (3.521 → 3.456) (2603.18534); the optimal synthetic share grows with model size,
**12.5% @3B → 25% @6B** (2509.15248). **Verdict (DERIVED):** our budget is in the regime where
textbook-alone hurts — **do not synthesize pretrain bulk**; spend it on the SFT/agent shell and the
verifier loop, as `2026-10-15-data-strategy.md` concluded.

## 5. Order and curriculum

14M-1B params / 300B tokens / three linguistic curricula vs random: all orderings share the same **latent
phase sequence**, curricula change **phase occupancy**; they reduce gradient noise scale and output-head
spectral saturation **up to 160M**, gains shrinking at 410M+; **reversing the order loses most of the
advantage** (2601.21698). 200+ models up to 100B tokens / 6 difficulty metrics / strict, paced and
**interleaved** strategies: advantages in the early and mid phases, some combos ~**6% above baseline**
(2506.11300). The decay phase is where the loss drops (2508.01483, 2410.05192) and the **annealing data is
named a first-order factor** (2507.17634). **DERIVED:** interleave within each stage, stage the mixture
*shape* over time, and don't spend GPU on difficulty-ordering — though our 57M *active* parameters make
the 410M+ reading ambiguous (**SPECULATION**).

## 6. Recommended mix plan for `b1`

Assumptions: one 19.7 GB filtered English-dominant web corpus (`AGENTS.md` §2.6); 5-20B **byte** budget =
0.25-1.01 passes; byte-level, vocab 256, d2048, max_iter 4 (`port/configs/b1.toml`); 0.40 GFLOP/byte.

### 6.1 The plan (fractions by stage, SmolLM2 shape)

| stage | share of budget | web | code | math | RU | source for the shape |
|---|---|---|---|---|---|---|
| **S1** | 0 → 70% | **90%** | **10%** | 0% | 0% | SmolLM2 S1 (10% code) 2502.02737; code-for-general-ability ICLR 2025 |
| **S2** | 70 → 90% | **80%** | **15%** | **5%** | 0% | SmolLM2 S2 (20% code, 5% OWM) 2502.02737; shipped code 13-17% (2407.21783, 2402.16819) |
| **S3** (decay) | 90 → 100% | **62%** | **20%** | **13%** | **5%** | SmolLM2 S4 (24% Stack-Edu, 14% math, 58% web) 2502.02737; decay-data-first 2507.17634 |

At 10B bytes: S1 7.0B (6.30B web / 0.70B code) · S2 2.0B (1.60 / 0.30 / 0.10) · S3 1.0B (0.62 / 0.20 /
0.13 / 0.05 RU). Web total 8.52B = **0.43 passes** — nowhere near the 4-epoch rule (2305.16264).

**Why not fit a law.** RegMix/DoReMi/CLIMB each cost proxy runs we cannot afford (2407.01492, 2504.13161)
and the 2026 evidence says the payoff is small (EACL 2026 survey; 2501.11747). Use the **free** proxies
instead: BiMix's entropy proxy is training-free (2405.14908), and one coarse 9.2M grid — web:code ∈
{90:10, 85:15} × math ∈ {0, 5, 10} — covers our decision space.

### 6.2 Acquire vs build from what we have

| need | acquire | build from the 19.7 GB? |
|---|---|---|
| **code ≈ 1.2 GB** | The Stack v2 permissive subset (2402.19173) — **licence filter mandatory**; or StarCoderData | **SPECULATION:** code-shaped docs exist in a web crawl and the in-tree DCLM classifier could be repurposed, but no threshold exists and it is unvalidated |
| **math ≈ 0.23 GB** | FineMath4+/3+ (ODC-By, 2502.02737 §3.3); InfiMM-WebMath | same caveat, unmeasured |
| **RU ≈ 0.05-0.5 GB** | FineWeb2 RU shard (ODC-By, 2506.20920) or CulturaX RU (2309.09400) | **no** — corpus is English-dominant; RU volume in it is unknown and unmeasured |
| **long documents** | books / scientific PDFs (2402.16819) | likely absent; **no source says long docs matter at our scale** — open question, not a purchase |
| **eval carves** | — | **carve code + math + RU tails first**; without them every claim above is unmeasurable (`AGENTS.md` §2.6) |

Total acquisition ≈ **1.5 GB** — trivial next to the 19.7 GB on disk, and it moves the mix more than any
ratio search would.

### 6.3 Order of operations

1. Carve **code + math + RU** eval tails and print the window; nothing below is measurable without it.
2. Acquire the ~1.5 GB; run exact dedup **only on the new data** (the filtered copy is already deduped).
3. 9.2M smoke, 6 cells, 3 seeds (`AGENTS.md` §1.2).
4. 5B-byte `b1` on the S1 shape; extend to 10B with S2/S3 only if the curve is still falling.
5. Keep **domain-level** BPB for code/math/RU on every eval line and use it — not a global epoch rule — to set those domains' repeat counts (2608.14071).

## VERIFIED (sourced)

Every number appears with its id in the tables above; none is a secondary summary. Methods: 2305.10429 ·
2407.01492 · 2403.16952 · 2405.14908 · 2312.02406 · 2504.13161 · ICLR 2026 FastMix · 2602.00747 ·
2603.28858 · ACL 2026 Data Mixing Agent · 2501.11747 · counter-evidence `2026.findings-eacl.15` +
2604.16380. Recipes: 2407.21783 · 2402.16819 · 2502.02737 · 2412.15115 · NVIDIA Nemotron-3 docs ·
2505.02881. Code: ICLR 2025 "To Code or Not To Code?" · 2409.04556 · 2610.11548. Multilingual: 2311.09205 ·
2510.25947 · 2510.22037 · 2504.04152. Epochs: 2305.16264 · 2511.13421 · 2608.14071. Synthetic: 2510.01631 ·
2603.18534 · 2509.15248. Order: 2601.21698 · 2506.11300 · 2508.01483 · 2410.05192 · 2507.17634.

## SPECULATION (my inference, marked inline above)

- **The stage fractions in §6.1** — no published mix exists for a byte-level 1B at 5-20B bytes.
- **"A fitted mixing law is not worth it for us"** — inferred from the EACL-2026 survey + MEDU, not a
  head-to-head at our scale.
- **Mining code/math subsets from the existing 46.2 GB raw** — plausible, never measured.
- **EN:RU ≈ 95:5 as a decay upsample** — 2510.25947 supports balance, gives no ratio at 1B.
- **Ordering/curriculum is not worth GPU at our scale** — effects shrink at 410M+, but our 57M *active*
  parameters make that reading ambiguous.
- **Long-document data may or may not matter for us** — no source found at our scale.

## Confidence

**HIGH**: method numbers/costs, shipped recipes, the code-beneficial / code-not-monotone pair, the epoch
findings, the synthetic verdict, the decay-phase lever. **MED**: the stage-fraction shape applied to a
byte-level model (SmolLM2 is BPE), the multilingual reading at our scale, any per-source token count.
**LOW / SPECULATION**: the §6.1 percentages, the RU ratio, and the "skip the fitted law" call.

## Open questions

1. **No published mixture for a byte-level 1B at 5-20B bytes**; SmolLM2 and BLT are the nearest.
2. **We have no code, math or RU eval carve** — §6.1 cannot be checked today; build it first.
3. **No head-to-head of "fit a mixing law" vs "static stage mix" at 1B/≤20B bytes.**
4. **Per-domain repetition limits are unmeasured for us** — 2608.14071 gives the procedure only.
5. **What is in our 19.7 GB?** No code/math/RU/length breakdown exists, and every acquire-vs-build call in
   §6.2 rests on that missing measurement.
