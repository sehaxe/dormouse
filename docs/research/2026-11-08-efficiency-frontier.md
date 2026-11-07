# Efficiency frontier: published loss-vs-compute points, the metric, and where a 95M byte model actually sits

**Date:** 2026-11-08. **Mode:** read-only research. No config, code, checkpoint or GPU was touched.
**Owner's claim under test:** "we must be more efficient than DeepSeek-V4.1 and the top models, not GPT-2."
**Question restated:** what would make that claim true or false, on a like-for-like axis, and what is the honest state at 95M / 0.25 GB / 1.46 BPB?

**Tags:** **[T]** read from a table, figure or section of the cited paper · **[C]** computed here, formula shown · **[S]** speculation, no direct measurement · **[ours]** measured in this repository.

**Citation rule.** Every paper number carries `id vX (first-posted date) §/Table`. Where a report publishes only benchmark scores, the row says **BENCHMARK-ONLY** and no loss is inferred. Where a quantity is not published, the cell says **NOT PUBLISHED** — it is not reconstructed from a figure unless the figure has numeric labels.

---

## TL;DR (RU)

1. **Ни одна из моделей, названных владельцем, не публикует таблицу loss-vs-compute.** V4.1-Flash — одна модель, 45T токенов, ≈2.16e24 FLOPs (prefill-базис), BPB только на рисунке 6, без GPU-часов, без токенизатора в отчёте и без кривой. K3 — рисунок 7 с осями 1e20–1e21 и подписью «2.5×», без таблицы. K2, V3, GLM-5, Qwen3 — бенчмарки, не loss. **Доля их опубликованной кривой ниже 1e20 FLOPs = 0.**
2. **Абсолютные точки с известным compute существуют у четырёх источников:** Cerebras-GPT (7 точек, nats/token, Pile test), Chinchilla (1 точка, BPB на Pile), Olmo Hybrid (7 точек, task-BPB), Qwen3.8-Next (1 ablation-точка, nats/token). Ещё две абсолютные BPB-точки есть в Table 3 DeepSeek-V3: сам V3 (0.548) и Llama-3.1-405B (0.542). Всё остальное — относительные кривые на рисунках.
3. **Наша точка — 1.42e17 FLOPs (6ND).** Ближайшая табличная точка выше — Cerebras-GPT 111M на 1.47e18 (в 10.3 раза больше счёта), 2.608 nats/token ≈ **0.94 BPB** при 4.0 байта/токен [C]. Это на 0.4–0.5 BPB лучше нашей 1.4621, но межпаперный разброс nats/token при тех же (N,D) достигает 0.8 нат (Chinchilla-закон предсказывает 3.43 там, где Cerebras измерил 2.608), то есть ≈0.3 BPB. Разрыв на грани объяснимого доменом — и он **не в нашу пользу**.
4. **Кривая в брифе смешана из двух прогонов.** `1.927 @ 98.3 MB` — это `logs/f3short.log:49`, прогон **до** фикса Muon (`0a7dfcd`, 2026-11-07 04:27); claims ledger относит такие `opt=mix`-числа к retracted class. `1.519 @ 245.76 MB` — `logs/flagship3.log:31`, **после** фикса. На одном прогоне: **1.660 @ 98.3 MB → 1.519 @ 245.76 MB**, наклон **−0.107 BPB/удвоение**. Смешанная пара даёт −0.307/удвоение, что при экстраполяции к 1e20 уводит ниже 0.5 BPB — ниже энтропии английского текста, то есть это не закон.
5. **Точка 1.4621 @ 0.25 GB не имеет артефакта в репозитории.** Я не нашёл ни лога, ни jsonl с этой тройкой (bpb, offset 100 000 000, 196 563 байта). Её окно eval — другое (`off=100M`), поэтому она **не лежит на кривой** 1.519 (`off=0`): разница 0.057 BPB объясняется трудностью окна, а не +1.6 % данных.
6. **«116M dense control @ 500 MB = 1.356» в репозитории не существует.** Документированный контроль — **13.5M, 1.492 @ 500 MB** (`docs/reviews/2026-11-06-control-transformer.md:6`). Ближайшее к «1.356» — литературный якорь 1.20–1.35 для 124M vanilla FineWeb-Edu на ~1B токенов (`2026-11-06-bpb-accounting.md:353`), это не измеренный прогон.
7. **«Более эффективны, чем V4.1» — не тестируемое утверждение при нашем масштабе.** У V4.1 нет ни одной точки ниже 1e23; ниже 1e20 прошло **0.0046 %** его токенов (2.08e9 из 45e12) и ни одного чекпойнта. Сравнивать можно только экстраполяцию их закона вниз на 7 порядков — а это ровно тот режим, где степенные законы наименее надёжны.
8. **Честное утверждение, которое можно проверить:** «на 1e19–1e20 FLOPs на Pile-test BPB наш закон имеет меньший intercept, чем Cerebras/Chinchilla». Цена: 760M×22GB = 1.00e20 (10.1 GPU-ч на 8×H100, ≈$160) или 3.3 года на этой карте при 2 % MFU. Самый дешёвый фальсифицируемый тест — **111M×2.2GB = 1.47e18** (17.8 дней локально или 14.7 GPU-ч на 8×H100, ≈$235) против строки Cerebras 2.608 nats/token.

---

## PART 1 — Published loss-vs-compute points

### 1.1 The named frontier models: what is actually published

| Model | Report (id v, date) | Total / active | Tokens | Compute (6ND, basis) | Loss published? | Metric / eval | Tokenizer | Match type | Status |
|---|---|---|---:|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 2609.19969v1 (2026-09-17) | 552B + 196B Engram / 8B prefill, 16B decode | 45T | **2.16e24** prefill; 4.32e24 decode [C] | **NO** — Fig. 6 BPB bars, no numeric labels in text | BPB on internal held-out corpora | 129,280 (HF config, not in report) | single model | **BENCHMARK-ONLY**; no GPU-hours, no curve |
| DeepSeek-V3 | 2412.19437v2 (2024-12-27) | 671B / 37B | 14.8T | **3.29e24** [C]; 2.664M H800-h pretrain (Table 1) | Pile-test **BPB 0.548** (Table 3) | BPB, Pile test | 128K BPE | single model | one absolute point at 3.3e24 |
| Kimi K3 | 2607.24653v2 (2026-07-27) | 2.78T / 104.2B | **NOT PUBLISHED** | not computable | **NO** — Fig. 7 fitted curves, axis 1e20–1e21, "2.5× vs K2" | validation loss vs FLOPs | 160K | iso-loss (fitted) | figure only; no table, no token count |
| Kimi K2 | 2507.20534v2 (2025-07-28) | 1.04T / 32.6B | 15.5T | **3.03e24** [C] | **NO** | — | 160K | single model | no loss axis |
| GLM-5 | 2602.15763v2 (2026-02-17) | 744B / 40B | 28.5T | **6.84e24** [C] | **NO** (only SFT loss A/B MLA vs DSA, Fig. 6) | — | 154,880 | single model | no GPU-hours, no curve |
| Qwen3-235B-A22B | 2505.09388v1 (2025-05-14) | 235B / 22B | 36T | **4.75e24** [C] | **NO** | — | 151,669 | single model | benchmark-only |
| Qwen3.8-Next | 2608.30320v1 (2026-08-31) | 125B / 6B + 51B n-gram | flagship **NOT PUBLISHED**; ablation 4T | ablation **2.14e22** [C] | ablation **1.5702 nats/token** (Fig. 8 text) | training loss, internal | not checked | iso-tokens ablation | the only tabulated loss in the 2026 set |
| Llama 3 405B | 2407.21783v3 (2024-07-31) | 405B dense | 15.6T | **3.79e25** (published 3.8e25) | Pile-test **BPB 0.542** via V3 Table 3 (same harness) | BPB, Pile test | 128K | single model | one absolute point at 3.8e25 |
| Llama 4 | no arXiv report | — | — | — | **NOT PUBLISHED** | — | — | — | model card/blog only |

Read that table as the answer to "why can't we just compare": **V4.1, K3, K2, GLM-5, Qwen3 and Qwen3.8 publish zero tabulated loss-vs-compute points**; V3 and Llama-3/4 contribute one absolute BPB benchmark point each (V3 and Llama-3.1-405B, both in V3's Table 3 under V3's harness, at 3.3e24 and 3.8e25), and Llama 4 publishes nothing.

### 1.2 The absolute points that do exist (all sources)

`BPB` is bits per **byte** of the scored text. `nats/token` is not BPB; the conversion needs bytes/token and is marked [C]. Cerebras and Qwen3.8 do not publish bytes/token, so their BPB column is indicative at 4.0 B/token, with ±0.05 BPB per 0.1 B/token.

| Source (id v, date) | Point (N, D) | Compute | Published loss | Normalised BPB [C] | Eval | Table |
|---|---|---:|---|---|---|---|
| Cerebras-GPT (2304.03208v1, 2023-04-06) | 111M, 2.2B | 1.47e18 | 2.608 nats/tok | 0.941 | Pile test | Table 3 |
| Cerebras-GPT | 256M, 5.1B | 7.83e18 | 2.349 | 0.847 | Pile test | Table 3 |
| Cerebras-GPT | 590M, 11.8B | 4.18e19 | 2.181 | 0.787 | Pile test | Table 3 |
| Cerebras-GPT | 1.3B, 26.3B | 2.05e20 | 1.997 | 0.720 | Pile test | Table 3 |
| Cerebras-GPT | 2.7B, 53.0B | 8.59e20 | 1.834 | 0.661 | Pile test | Table 3 |
| Cerebras-GPT | 6.7B, 133.2B | 5.35e21 | 1.704 | 0.615 | Pile test | Table 3 |
| Cerebras-GPT | 13B, 257.1B | 2.01e22 | 1.572 | 0.567 | Pile test | Table 3 |
| Olmo Hybrid (2604.03444v4, 2026-04-03) | 73M, 11.7B | 5.12e18 | **1.077** | 1.077 | OlmoBaseEval Easy (Math/Code/QA task BPB) | Table 5 |
| Olmo Hybrid | 130M, 20.8B | 1.62e19 | 0.986 | 0.986 | same | Table 5 |
| Olmo Hybrid | 254M, 40.6B | 6.19e19 | 0.891 | 0.891 | same | Table 5 |
| Olmo Hybrid | 523M, 83.7B | 2.63e20 | 0.799 | 0.799 | same | Table 5 |
| Olmo Hybrid | 722M, 115.5B | 5.00e20 | 0.754 | 0.754 | same | Table 5 |
| Olmo Hybrid | 948M, 151.7B | 8.63e20 | 0.717 | 0.717 | same | Table 5 |
| Olmo Hybrid | 1482M, 237.1B | 2.11e21 | 0.669 | 0.669 | same | Table 5 |
| Chinchilla (2203.15556v1, 2022-03-29) | 70B, 1.4T | 5.88e23 | 0.667 (pile_cc); 0.647 (openwebtext2) | BPB as published | Pile subsets | Table A5 |
| DeepSeek-V3 (2412.19437v2) | 671B/37B, 14.8T | 3.29e24 | 0.548 | BPB as published | Pile test | Table 3 |
| Llama-3.1-405B (via 2412.19437v2) | 405B, 15.6T | 3.79e25 | 0.542 | BPB as published | Pile test, same harness | Table 3 |
| Qwen3.8-Next (2608.30320v1) | 10.8B-A0.89B, 4T | 2.14e22 | **1.5702 nats/tok** | — (needs B/tok) | internal training loss | Fig. 8 text |
| **ours** | 95.0M, 0.25e9 B | **1.42e17** | **1.4621 BPB** | 1.4621 | fwedu, off=100M, 196,563 B | owner-supplied (see §3.2) |

Two structural facts fall out of this table:

1. **The table has a 10× hole between 1.47e18 and 5.12e18, and a 2.6× hole between 5.12e18 and 1.62e19.** Cerebras and Olmo Hybrid are the only sources that fill the 1e18–1e21 band with actual rows. The frontier models sit four to six orders of magnitude above it.
2. **The metrics are not the same quantity.** Cerebras is nats/token on raw Pile text. Olmo Hybrid is *task* BPB on Math/Code/QA answers. Chinchilla/V3/Llama are raw-text BPB on Pile. Our number is raw-text BPB on FineWeb-Edu. Mixing them in one plot is invalid without the corrections in Part 2.

### 1.3 Rejected candidates (examined, not usable)

| Candidate | Why rejected |
|---|---|
| DeepSeek-V4.1-Flash | No loss axis: Fig. 6 is BPB bars on internal corpora, no numeric labels in the text, no GPU-hours, no tokenizer in the report. Cannot yield a (compute, loss) point. |
| Kimi K3 | Fig. 7 only; axes 1e20–1e21; the "2.5×" is an aggregate horizontal shift, not a tabulated point; token count not published. |
| Kimi K2 | 15.5T tokens and 32B active give a single compute number; no loss table. |
| GLM-5 | 28.5T tokens; only an SFT loss A/B (MLA vs DSA, Fig. 6), which is a method comparison at fixed scale, not a compute curve. |
| Qwen3 | 36T tokens; benchmark tables only. |
| Llama 3 | IsoFLOP study at 6e18–1e22 exists, but the paper shows it in Figures 2–3 and never tabulates the points; the only absolute number is the 3.8e25 FLOPs and 15.6T tokens of the 405B model. |
| Llama 4 | No arXiv report; model card/blog. |
| OLMoE | Figure 2 loss curves, no BPB table; the "~3× fewer FLOPs" is a curve comparison. |
| OLMo 3 | Figure 6 BPB vs compute 1e19–1e25, no table. |
| Pythia | Loss curves in Figure 2; Table 3 gives GPU counts, not loss. Its checkpoints make any point derivable, which is why it is the standard small-scale anchor — but the paper itself publishes no (C, L) row. |
| Kaplan | Figure 1 only; the paper tabulates the fit (Table 2), not the points. |
| TinyStories | 1M–33M params, synthetic children's stories, loss in figures; no table and no token count in the read sections. It is the only family at 1e16–1e17, and it cannot anchor an efficiency claim about English web text. |
| MiniCPM, SmolLM2, DCLM | Benchmark tables and data ablations; no absolute loss-vs-compute row. DCLM Table 33 is benchmark scores only. |

---

## PART 2 — The metric, defined once

### 2.1 Definitions

**Training compute.** For a dense transformer, `C ≈ 6·N·D` forward+backward, where `N` is non-embedding parameters and `D` is tokens (Kaplan 2001.08361v1 §2.1; Chinchilla 2203.15556v1 §3 uses `FLOPs(N,D) ≈ 6ND`). For an MoE, `N` must be the **active** parameter count per token, not total: V3 is 671B total but 37B active, so its 14.8T-token run is 3.29e24, not 5.96e25. Attention and the output head are outside `6ND`; see §2.4.

**Bits per byte.** `BPB = −ln P(target | context) / (ln 2 · bytes(target))`, summed over the scored text. The reference definition is Gao et al.'s, restated by Olmo 3 2512.13961v2 §3.3.2: "the negative log-likelihood of the answer divided by the number of UTF-8 bytes in the answer string." BPB is tokenizer-invariant **only if** the bytes are the original text bytes and the harness detokenises exactly; it is not invariant if the harness scores token strings or counts special tokens.

**Why BPB and not nats/token.** `BPB = (nats/token) / (ln 2 · bytes/token)`. A tokenizer with 4.0 bytes/token and one with 3.3 bytes/token differ by 21 % in this divisor at the same nats/token. DeepSeek-V3 states the reason explicitly (2412.19437v2 §4.4.1): "use Bits-Per-Byte (BPB) as the metric to guarantee fair comparison among models using different tokenizers."

**Our case is the easy one.** `flagship3_dense` has vocab 256 and consumes raw bytes: 1 token = 1 byte, so `BPB = nats/token / ln 2` exactly, and `D` in `6ND` is bytes. No tokenizer correction exists or is needed.

### 2.2 What makes a comparison invalid, and the correction

| Invalid comparison | Why it breaks | Correction | Residual error |
|---|---|---|---|
| Different tokenizer, loss in nats/token | Same text becomes a different number of tokens; 3.3 vs 4.0 B/token is a 21 % divisor shift | Convert both to BPB over UTF-8 bytes | none if byte accounting is exact; ±0.02 BPB if special tokens are counted differently |
| Different data mixture | Loss is a property of (model, distribution), not model alone; a code-heavy model pays on prose and vice versa | Evaluate both on one fixed public set (Pile test) and report the in-domain set separately | the domain gap is real, not noise: our own repo measured **+0.36 BPB** cross-corpus (`2026-11-06-datapath-audit.md:154`) |
| Different eval set / split | Pile test vs fwedu vs OlmoBaseEval are different entropies | one shared set; never compare a raw-text BPB with a task-answer BPB | Olmo Hybrid's 1.077 is task BPB; it is not comparable to a raw-text 1.077 |
| Different context length | Loss at 1K vs 128K differs; long-context models are trained on a different objective mix | evaluate at the same context, or state the context | unknown without a sweep; not quantifiable from these papers |
| Params-matched instead of compute-matched | At fixed N, more D is more compute; at fixed D, more N is more compute | report `C=6ND` and label the match (iso-param, iso-token, iso-FLOP, iso-loss) | none if labelled; the labels are the whole claim |
| Compute counted on total instead of active params (MoE) | V3 total 671B vs active 37B is an 18× error | use active params; include shared expert and router (small) | <2 % for these models |
| Precision-weighted FLOPs | V4.1 itself weights BF16/FP8/FP4 as 1/0.5/… (2609.19969v1 §2), so "FLOPs" across precisions are not the same unit | compare within one precision, or state the weight | not resolvable across papers |

### 2.3 The cross-paper spread is the size of the effect we are arguing about

At (N=111M, D=2.2B), the Chinchilla fitted law gives

`L = 1.69 + 406.4/111e6^0.34 + 410.7/2.2e9^0.28 = 3.43 nats/token` [C],

while Cerebras-GPT measured **2.608 nats/token** at the same (N, D) on the Pile (Table 3). That is a **0.82-nat** spread — ≈0.30 BPB at 4.0 B/token — produced entirely by data, eval and tokenizer differences. Any claimed efficiency edge smaller than ~0.3 BPB between two different corpora is inside this spread and is not evidence.

This is the single most important number in Part 2: our 1.4621 vs Cerebras' ≈0.94 is a 0.52 BPB gap, of which ~0.3 BPB could be corpus/eval. The remaining ~0.2 BPB is unexplained and points the wrong way.

### 2.4 What `6ND` does not count

- **Attention.** At 95M, 12 layers, d=832, s=1024, the attention core is **122.7 MFLOP/byte** on top of `6N = 570.2`, i.e. **+17.7 %** (`docs/research/2026-11-07-frontier-vs-vanilla.md:133-138`). So our true training bill at 250 MB is **1.68e17**, not 1.42e17. Every paper number in Part 1 also excludes attention; the comparison is consistent only if both sides use `6ND`. State the basis.
- **Embeddings and output head.** Kaplan excludes them by definition. At vocab 256 they are negligible for us (<1 %). At vocab 100k–160k and N≈100M they are not: the head alone is `2·d·V` FLOP/token forward, which for d=384, V=100k is ≈77 MFLOP/token, +17 % on top of `6N=438`. This is a second reason 100M-class cross-paper compute is fuzzy.
- **Router/gates and shared experts** in MoE: small (<2 %) but should be named.

### 2.5 The only comparison shape that survives

`same eval set × same byte accounting × known active-param compute × labelled match type`. There are exactly two such comparisons in the literature read here:

1. **V3 vs Llama-3.1-405B** in V3 Table 3: 0.548 vs 0.542 BPB on Pile test, at 3.29e24 vs 3.79e25 FLOPs — **11.5× less compute for +0.006 BPB**. This is a real frontier efficiency claim, published, same harness, same metric. It is also at 1e24–1e25.
2. **Chinchilla vs Gopher** in Table A5: 0.667 vs 0.691 on pile_cc at 5.88e23 vs 5.76e23 — a **compute-matched** win, +0.024 BPB.

Everything else in Part 1 is either within-paper relative (K3 2.5×, OLMoE 3×, Qwen3.8 9×) or a benchmark-equivalence claim, not a loss-vs-compute point.

---

## PART 3 — Where our point sits

### 3.1 The points, recomputed

`C = 6·N·D` with N = 95,035,200 and D = bytes seen (`bytes_per_step = 24,576`, so step × 24,576 B). `C_attn` adds the 17.7 % attention term. "0.25 GB" is read as 250e6 bytes, which reproduces the brief's 1.42e17; if it means GiB (268.4e6), C = 1.53e17 (+7 %), which does not change any verdict here.

| Claimed point | Artifact | Bytes seen | C = 6ND | C with attention | BPB | Eval window | Status |
|---|---|---:|---:|---:|---:|---|---|
| 1.927 @ 98 MB | `logs/f3short.log:49` (step 4000) | 98,304,000 | 5.60e16 | 6.59e16 | 1.927 | off=0, 196,608 B | **RETRACTED class** — run is pre-Muon-fix (mtime 2026-11-06 22:11 < `0a7dfcd` 2026-11-07 04:27) |
| 1.660 @ 98 MB | `logs/flagship3.log:15` (step 4000) | 98,304,000 | 5.60e16 | 6.59e16 | 1.660 | off=0, 196,608 B | valid, post-fix |
| 1.519 @ 246 MB | `logs/flagship3.log:31` (step 10000) | 245,760,000 | 1.40e17 | 1.65e17 | 1.519 | off=0, 196,608 B | valid, post-fix |
| 1.4621 @ 250 MB | **no artifact found** (owner-supplied) | 250,000,000 | 1.42e17 | 1.68e17 | 1.4621 | off=100,000,000, 196,563 B | **UNVERIFIED ARTIFACT**; different eval window |
| 1.492 @ 500 MB (control) | `docs/reviews/2026-11-06-control-transformer.md:6` | 500,000,000 | 4.04e16 | — | 1.492 | same harness | valid, but N=13,478,752 dense, not our model |
| "1.356 @ 500 MB, 116M" | **no artifact** | — | 3.48e17 if real | — | 1.356 claimed | — | **NOT FOUND** |

### 3.2 Three integrity corrections, before any plotting

**(a) The 98 MB and 246 MB points are from different runs.** `f3short` ran on 2026-11-06 22:11, before the Muon LR seam was fixed (`0a7dfcd`, 2026-11-07 04:27). `flagship3` ran at 07:16 the next morning. The two run lines differ only in `steps` (4000 vs 40000); both are `flagship3_dense`, 95,035,200 params, 24,576 B/step, lr 3e-4, `opt=mix`, `grad_accum=3`, and both eval the same `off=0` window. The fix moved the 98.3 MB point **1.927 → 1.660 (−0.267 BPB)**. No config snapshot exists for either short run, so "the fix alone" is an inference from identical run lines plus the one commit between them, not a controlled A/B; the step-500 train loss (2.157 vs 1.394) is the second line of evidence. The claims ledger's retraction rule explicitly covers "a BPB/Δ claim from an `opt=mix` JSONL whose mtime is before `0a7dfcd`". Therefore **1.927 is not a point of the current model**. The same-run curve is 1.660 → 1.519.

**(b) The 0.25 GB point is not on the 1.519 curve.** It scores a different window (`off=100,000,000`, 196,563 B) than 1.519 (`off=0`, 196,608 B). The 0.057 BPB difference is the window, not the +1.7 % data. To put them on one axis, the 0.25 GB run must be re-scored at `off=0`, or the 246 MB checkpoint re-scored at `off=100M`. Until then they are two different measurements.

**(c) "116M / 1.356" is not in the repository.** The documented dense control is 13,478,752 params, 1.492 at 500 MB (`control-transformer.md:6,271`). The 1.20–1.35 figure is a literature anchor for a 124M vanilla FineWeb-Edu at ~1B tokens (`2026-11-06-bpb-accounting.md:353`), not a measured 500 MB run. If a 116M/1.356 artifact exists outside this repo, it must be produced before it is used: at 2.4× our compute it would imply a −0.106 BPB gap at −0.081 BPB/doubling, which is exactly the slope the valid 95M run shows — plausible, but unverified.

### 3.3 Placement against the frontier

Order of magnitude: **ours 1e17 → nearest tabulated point 1.5e18 → first points above 3e19: Pythia-70M at 3.4e19 (no table), Olmo Hybrid 254M at 6.2e19 (task BPB) → first frontier-lab point 3.3e24.** The gap between our point and V4.1 is seven orders of magnitude; the gap to the nearest tabulated row is one order.

| Model / point | Compute | BPB | Gap to us (compute) | Comparable metric? |
|---|---:|---:|---:|---|
| **ours (owner point)** | **1.42e17** | **1.4621** | 1× | raw-text BPB, fwedu |
| Cerebras-GPT 111M | 1.47e18 | ≈0.94 [C] | **10.3×** | raw-text nats/token → BPB, Pile |
| Olmo Hybrid 73M | 5.12e18 | 1.077 | 36× | task BPB, not raw text |
| Olmo Hybrid 254M | 6.19e19 | 0.891 | 436× | task BPB |
| Pythia-70M endpoint | 3.40e19 (non-emb; 1.26e20 on total params) | figure only | 240× | raw-text loss curve, no table |
| Cerebras-GPT 1.3B | 2.05e20 | ≈0.72 [C] | 1444× | raw-text, Pile |
| Olmo Hybrid 1.48B | 2.11e21 | 0.669 | 14,859× | task BPB |
| Qwen3.8 ablation | 2.14e22 | 1.5702 nats/tok | 150,704× | internal training loss |
| Chinchilla 70B | 5.88e23 | 0.667 (pile_cc) | 4.14e6× | raw-text BPB, Pile |
| DeepSeek-V3 | 3.29e24 | 0.548 | 2.32e7× | raw-text BPB, Pile |
| Llama-3.1-405B | 3.79e25 | 0.542 | 2.67e8× | raw-text BPB, Pile |

**No frontier point overlaps our compute range.** The nearest tabulated point is Cerebras 111M at 10.3× our compute. The nearest *any* model family is TinyStories (1M–33M params), but its domain is synthetic children's stories and it publishes no table — it cannot anchor a claim about English web.

**Overlap starts at ~1.5e18** (Cerebras 111M), and the first *raw-text BPB table row* at 1.47e18 is the same point. The first frontier-lab point is at 3.3e24 — seven orders of magnitude above us.

### 3.4 The run that reaches the frontier band

- **760M × 22GB = 1.00e20** — the cheapest run that enters the published 1e20 band, using the whole corpus at 232 tokens/param.
- **1B × 22GB = 1.32e20** — the natural next rung, 287 tokens/param.
- At 95M, reaching 1e20 needs **175 GB** of data (8× the corpus); at 1B it needs the 22 GB we already have.

Cost at our measured 0.954 TFLOP/s effective (47.7 TFLOP/s card × 2 % MFU): 1.00e20 / 0.954e12 = **1.05e8 s = 1,216 days (3.3 years)** for the 760M run; 1,600 days for 1B. At 8×H100, 35 % MFU, $2/GPU-h (V3's own rental assumption): **10.1 h ≈ $161** for 760M, **13.2 h ≈ $212** for 1B. The measurement is cheap rented and impossible locally — that asymmetry is the whole argument for renting it.

---

## PART 4 — Is "more efficient than V4.1" a testable claim?

### 4.1 Arithmetic: the fraction of their curve below 1e20

`D_below = 1e20 / (6·N_active)`. For V4.1 prefill N=8e9: **2.08e9 tokens**. Against 45e12 tokens that is **0.0046 %** of the run, and **zero released checkpoints** — the model is one endpoint, not a curve.

| Model | Tokens | Active N | Compute | Tokens below 1e20 | % of run | Checkpoints below 1e20 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 14.8T | 37B | 3.29e24 | 4.50e8 | 0.0030 % | 0 |
| DeepSeek-V4.1-Flash | 45T | 8B (prefill) | 2.16e24 | 2.08e9 | 0.0046 % | 0 |
| Kimi K2 | 15.5T | 32.6B | 3.03e24 | 5.11e8 | 0.0033 % | 0 |
| GLM-5 | 28.5T | 40B | 6.84e24 | 4.17e8 | 0.0015 % | 0 |
| **ours** | 2.5e8 | 95M | 1.42e17 | 2.5e8 | **100 %** | 1 |

So the answer to "what fraction of their learning curve is below 1e20" is: **0 % of published checkpoints; 0.0015–0.0046 % of tokens; and for K3, 0 % of the fitted curve's x-range, which starts at 1e20.**

### 4.2 What benchmark would have to be shared

**Pile-test BPB** is the only raw-text yardstick that ≥3 of these sources report: Chinchilla Table A5, Cerebras Table 3, DeepSeek-V3 Table 3 (which also carries Llama-3.1-405B). It is public, the test split is fixed, and BPB is already per byte. The correction is that our model must be evaluated on Pile test, not fwedu, and that all sides must state `6ND` vs attention-inclusive FLOPs.

Second choice: **OlmoBaseEval Easy** (public harness, task BPB), because Olmo Hybrid and OLMo 3 publish it. It is a task-answer metric, not raw text, so it answers "quality per FLOP on tasks", not "compression of English".

A benchmark score (MMLU, GSM8K) cannot substitute: the brief's rule is right — do not convert scores into loss.

### 4.3 Three candidate claims, ranked by defensibility

| Rank | Claim | Measurement that supports it | Measurement that kills it | Cost | Why this rank |
|---|---|---|---|---|---|
| **1** | "At 1.47e18 FLOPs, a 111M byte-level model beats Cerebras-GPT 111M's **2.608 nats/token** on Pile test." | 111M × 2.2GB = 1.47e18; Pile-test BPB < 2.608 nats (≈0.94 BPB @4.0) | Pile-test BPB > 2.608 nats | 17.8 days local, or 14.7 GPU-h on 8×H100 (~$235) | One exact published row, same N, same compute; the only cheap falsifiable claim |
| **2** | "At 1e20, our scaling law's intercept is below the Cerebras/Chinchilla law on Pile test." | 760M × 22GB = 1.00e20; Pile-test BPB < 0.94 (law value 2.605 nats at 4.0 B/tok; <1.02 if the true ratio is 3.7) | Pile-test BPB > 0.94 at 4.0 B/tok | 3.3 years local, or 10.1 GPU-h on 8×H100 (~$161) | Enters the published 1e20 band; still one point, not a law |
| **3** | "Our law has a lower intercept than V4.1 at equal compute." | **No measurement exists**: V4.1 has no point below 1e23, no curve, no shared eval | — (not falsifiable) | undefined | **Cannot be made at our scale** — see §4.4 |

A fourth claim, "we beat GPT-2 small at equal compute", is probably true (GPT-2 small at 1.4e17 has seen only 0.19 GB) and worth nothing: it is the comparison the owner already rejected.

### 4.4 Blunt verdict

**The claim "we are more efficient than DeepSeek-V4.1" cannot be made at our scale, and it is not a matter of more data or a better plot.** The arithmetic:

- V4.1's only published model is at **2.16e24** FLOPs (prefill basis). That is **1.5e7×** our compute.
- Its published BPB is a bar chart on internal corpora with no compute axis. There is no V4.1 point at 1e20, 1e21 or 1e22 to compare against.
- The fraction of its run below 1e20 is **0.0046 %** of tokens and **zero** checkpoints.
- Comparing our 1e17 point to its 2e24 point is a comparison of two different questions: "how well does a small model compress a small corpus" vs "how well does a 552B MoE compress 45T tokens". No rescaling makes them the same question without a shared law and a shared eval, and V4.1 publishes neither.

**What is true instead:** we have one measured point at 1.4e17 on English web, and the nearest published tabulated point (Cerebras 111M, 10.3× our compute) is ~0.4–0.5 BPB better, of which ~0.3 BPB could be corpus/eval. That is not evidence of an advantage; it is a warning that the advantage has not been demonstrated even against a 2023 dense baseline, let alone a 2026 frontier model.

**The strongest honest claim available today** is narrower and cheaper: *"at 1.47e18 FLOPs, on Pile-test BPB, our recipe beats the published Cerebras-GPT 111M row."* It is falsifiable in 18 local days or 15 rented GPU-hours. Until that row exists, "more efficient than the top models" is a statement about intention, not measurement.

---

## PART 5 — What the frontier's efficiency actually comes from

| Lever | The number in their report | Source | Transferable to 95M–1B on one card? | Why |
|---|---|---|---|---|
| **MoE active fraction** | V4.1 8B prefill / 552B = **1.4 %**; 16B decode = 2.9 %. K3 104.2B / 2.78T = **3.7 %**. V3 37B / 671B = **5.5 %**. K2 32.6B / 1.04T = **3.1 %**. GLM-5 40B / 744B = **5.4 %** | 2609.19969v1 abstract; 2607.24653v2 Table 1; 2412.19437v2 Table 1; 2507.20534v2 Table 1; 2602.15763v2 Table 10 | **Capacity: yes. Speed: no.** | At 95M the step is launch-bound (2 % MFU); every extra expert adds launches. The equal-quality FLOPs evidence for MoE starts at 1.3B active (OLMoE) and 2B (DeepSeekMoE 0.175×), not at 95M. Our own frontier teardown ranks MoE as the only bet worth a ~31 GPU-h A/B. |
| **Data quantity** | V3 14.8T; V4.1 45T; K2 15.5T; GLM-5 28.5T; Qwen3 36T. Tokens/param: V3 22k, V4.1 81k, GLM-5 38k | same reports | **Partially.** | Our whole corpus is 22 GB = 232 tokens/param at 95M. The frontier's volume is 1,000–2,000× larger in tokens; the transferable lesson is "overtrain small models", not "use 45T tokens". Our measured marginal is ≈0.17 BPB per doubling over 0.05–0.25 GB (`2026-10-18-data-capacity.md:18`), so data is still the largest live lever at our scale. |
| **LR / batch schedule** | K3: cosine beats WSD under **independently tuned** hyperparameters (2607.24653v2 §3.2). Qwen3.8: refit LR/batch law gives **−0.0072 nats** at 10.8B-A0.89B / 4T (2608.30320v1 §3.2, Fig. 8). GLM-5: cosine, warmup 0→2e-4, decay to 4e-5 (2602.15763v2 §3.2) | cited | **Yes — cheapest transferable lever.** | No extra FLOPs, no new hardware. Our own Muon fix bought **−0.267 BPB** at 98 MB — larger than any architecture delta measured in this repo. That is the scale of "efficiency" that is actually available to us, and it is recipe, not architecture. |
| **Precision** | V3 FP8: relative loss error **<0.25 %** vs BF16 (2412.19437v2 §3.3). K3 MXFP8 expert activations, MXFP4 QAT post-training (2607.24653v2 §4.1.4). V4.1 FP4 main KV (2609.19969v1 §2.4.4) | cited | **Partly.** bf16/f16 now compute on this box at **44 TFLOP/s** (AGENTS.md §2.1, measured 2026-10-10), so half precision is a live lever; **FP8/FP4 training precision is not implemented**. | Precision buys throughput and HBM, not loss per FLOP; V3's own FP8 run is within 0.25 % of its BF16 loss, i.e. a null quality effect. It changes the wall-clock denominator, not the curve. |
| **MFU** | V3 **2.664M H800-h** pretrain for 3.29e24 FLOPs ⇒ **34.6 % of BF16 peak** (989 TFLOPS) or **17.3 % of FP8 peak** (1979 TFLOPS) [C]. Ours: **2 %** of 47.7 TFLOPS | 2412.19437v2 Table 1; `2026-11-05-small-model-mfu.md` / `frontier-vs-vanilla.md:112` | **No.** | 2,048-GPU cluster, DualPipe, FP8 GEMM. And MFU is not a loss-efficiency metric: it moves wall-clock, not BPB at fixed FLOPs. Conflating the two is the main error in the owner's framing. |
| **"Scaling efficiency" claims** | K3 **2.5×** vs K2 (Fig. 7, fitted, 1e20–1e21). Qwen3.8 **≈9×** vs 397B-A17B = 3× fewer active params × 3× fewer tokens (2608.30320v1 abstract). OLMoE **~3×** vs dense at 1.3B active (2409.02060v2 Fig. 2) | cited | **No.** | None is decomposed into architecture vs data vs hyperparameters. Qwen3.8's 9× is arithmetic from 3×3, i.e. a benchmark-equivalence claim, not a lower loss-vs-compute curve. K3's 2.5× is an aggregate horizontal shift with no table. |

**Verdict for Part 5 (RU).** Единственные переносимые рычаги на 95M–1B — **данные и LR/batch-рецепт**; MoE — единственная архитектурная ставка с измеренным equal-quality FLOPs-выигрышем вблизи нашего масштаба, но на 95M она не измерена и упирается в launch-bound шаг. Precision переносится частично (bf16/f16 уже считают на этой карте, FP8/FP4 — нет), MFU не переносится вообще и к loss-vs-compute отношения не имеет. «2.5×» и «9×» — это агрегаты, а не разложенные механизмы; переносить их на 95M нельзя.

---

## VERIFIED (sourced)

1. **No named frontier report publishes a loss-vs-compute table.** V4.1: Fig. 6 BPB bars, no numeric labels in text; no GPU-hours; tokenizer not in report (129,280 only in the HF config, per `2026-11-07-frontier-teardown.md`). K3: Fig. 7 only, axis 1e20–1e21, no table, token count not published. K2/V3/GLM-5/Qwen3: benchmark tables only. [T]
2. **V3 and Llama-3.1-405B are the only absolute frontier BPB points:** 0.548 at 3.29e24 and 0.542 at 3.79e25, same harness (V3 Table 3). [T]
3. **Chinchilla Table A5** gives 0.667 (pile_cc) and 0.647 (openwebtext2) at 5.88e23 FLOPs, 70B/1.4T. [T]
4. **Cerebras Table 3** gives 7 Pile-test nats/token points from 111M/2.2B (2.608) to 13B/257.1B (1.572), with Eq. 1: `L(f) = (f/5.984e22)^−0.0737 + 0.5066`. [T]
5. **Olmo Hybrid Table 5** gives 7 task-BPB points from 73M (1.077) to 1482M (0.669), at 160 tokens/param. [T]
6. **Qwen3.8 Fig. 8 text** gives 1.5702 nats/token at 10.8B-A0.89B / 4T tokens; the "1/9 FLOPs" is 1/3 active params × 1/3 tokens. [T]
7. **K3 2.5× is a fitted figure**, not a tabulated point. [T]
8. **Our compute:** 95,035,200 params × 250e6 bytes × 6 = **1.42e17** FLOPs; with attention **1.68e17**. [ours/C]
9. **Our valid same-run curve:** `logs/flagship3.log` 1.660 at step 4000 (98.3 MB) → 1.519 at step 10000 (245.76 MB); slope **−0.107 BPB/doubling**. [ours]
10. **The 1.927 point is from `logs/f3short.log`**, mtime 2026-11-06 22:11, before the Muon fix `0a7dfcd` at 2026-11-07 04:27; the claims ledger retracts that class. [ours]
11. **The 1.4621 point has no artifact in the repo**, and its eval window differs (off=100M vs off=0). [ours]
12. **No 116M/1.356 run exists in the repo**; the documented control is 13.5M / 1.492 at 500 MB. [ours]
13. **Fraction of frontier tokens below 1e20:** V3 0.0030 %, V4.1 0.0046 %, K2 0.0033 %, GLM-5 0.0015 %; zero checkpoints. [C]

## SPECULATION (my inference)

1. **SPECULATION:** the 0.4–0.5 BPB gap to Cerebras 111M is partly domain, partly a real deficit. No shared-eval run exists; the estimate assumes 4.0 bytes/token for the GPT-2 tokenizer on the Pile and that the ~0.3 BPB cross-paper spread applies. Confidence: low-medium.
2. **SPECULATION:** a 760M×22GB run would land near 1.0 BPB on fwedu, based on our measured 0.17 BPB/doubling at 0.05–0.25 GB. The slope is measured over 2.3 doublings only; the extrapolation over 9.5 doublings is not a law. Confidence: low.
3. **SPECULATION:** an 111M×2.2GB run would beat Cerebras' 2.608 nats/token on Pile test, because our byte-level pipeline avoids the tokenizer/head overhead and our data is cleaner. No measurement exists. Confidence: low-medium.
4. **SPECULATION:** part of the frontier's "2.5×/9×" is hyperparameter and data work, not architecture, because neither report decomposes the gain and our own −0.267 BPB came from an optimizer fix. Confidence: medium.

## Open questions

1. Where is the artifact for 1.4621 / off=100M / 196,563 B? Until it is named, the point is not reproducible.
2. Does a 116M/1.356 run exist outside this repo? If yes, it is the closest thing to a compute-matched dense control and must be re-scored at off=0.
3. What is K3's pretraining token count? Without it, its compute is not computable and the 2.5× cannot be placed on an absolute axis.
4. What are V4.1's GPU-hours and MFU? The report gives neither; the 2.16e24 is our 6ND estimate, not their number.
5. Which exact BPB values are behind V4.1 Fig. 6? The bars have no numeric labels in the text layer; reading pixels would be an inference, not a table read.

## Sources

| Id | Version / first posted | Used for |
|---|---|---|
| 2001.08361 | v1, 2020-01-23 | Kaplan `C ≈ 6NBS`, `αC=0.050`, `Cc=3.1e8` PF-days, 50,257 BPE |
| 2203.15556 | v1, 2022-03-29 | Chinchilla `6ND`, Table 3 compute-optimal, Table A5 BPB, 32k SentencePiece |
| 2304.01373 | v2, 2023-04-03 | Pythia sizes, 300B tokens, Table 3 GPU counts |
| 2304.03208 | v1, 2023-04-06 | Cerebras Table 1/3, Eq. 1, GPT-2 50,257 |
| 2305.07759 | v2, 2023-05-12 | TinyStories sizes, figures only |
| 2406.11794 | v4, 2024-06-17 | DCLM Table 1/33, benchmark-only rejection |
| 2407.21783 | v3, 2024-07-31 | Llama 3 405B, 15.6T, 3.8e25, 128K, isoFLOP range 6e18–1e22 |
| 2409.02060 | v2, 2024-09-03 | OLMoE 6.9B/1.3B, 5.1T, Fig. 2 |
| 2412.04403 | v2, 2024-12-05 | OLMo 2 ladder Table 1, task-loss fits |
| 2412.19437 | v2, 2024-12-27 | V3 Table 1/3, 2.664M H800-h, 128K, FP8 |
| 2502.02737 | v1, 2025-02-04 | SmolLM2, benchmark-only rejection |
| 2505.09388 | v1, 2025-05-14 | Qwen3 36T, 151,669, 235B-A22B |
| 2507.20534 | v2, 2025-07-28 | K2 1.04T/32.6B, 15.5T, 160K |
| 2512.13961 | v2, 2025-12-15 | Olmo 3 BPB definition and Fig. 6 range |
| 2602.15763 | v2, 2026-02-17 | GLM-5 744B/40B, 28.5T, 154,880, cosine |
| 2604.03444 | v4, 2026-04-03 | Olmo Hybrid Table 5/22, 160 tokens/param |
| 2607.24653 | v2, 2026-07-27 | K3 2.78T/104.2B, Fig. 7, 160K, cosine>WSD, MXFP8/FP4 |
| 2608.30320 | v1, 2026-08-31 | Qwen3.8 125B/6B, 9× claim, 1.5702 nats at 2.14e22 |
| 2609.19969 | v1, 2026-09-17 | V4.1 552B/8B/16B, 45T, Fig. 6, FP4 KV |

## Appendix A — arithmetic

```
ours:            6 × 95,035,200 × 250,000,000       = 1.4255e17
ours @ 98.3MB:   6 × 95,035,200 × 98,304,000        = 5.6034e16
ours @ 245.76MB: 6 × 95,035,200 × 245,760,000       = 1.4009e17
attention:       12 × 12 × 832 × 1024               = 1.2266e8 FLOP/byte
6N:              6 × 95,035,200                     = 5.7021e8 FLOP/byte
attention share: 1.2266e8 / (5.7021e8 + 1.2266e8)   = 17.7 %
V3:              6 × 37e9 × 14.8e12                  = 3.2856e24
V4.1 prefill:    6 × 8e9 × 45e12                    = 2.1600e24
V4.1 decode:     6 × 16e9 × 45e12                   = 4.3200e24
K2:              6 × 32.6e9 × 15.5e12               = 3.0318e24
GLM-5:           6 × 40e9 × 28.5e12                 = 6.8400e24
Qwen3:           6 × 22e9 × 36e12                   = 4.7520e24
Llama3-405B:     6 × 405e9 × 15.6e12                = 3.7908e25
Cerebras 111M:   6 × 111e6 × 2.2e9                  = 1.4652e18
OlmoH 73M:       6 × 73e6 × 11.68e9                 = 5.116e18
Pythia-70M:      6 × 70e6 × 300e9 (total params)     = 1.260e20
Pythia-70M:      6 × 18.915e6 × 300e9 (non-emb)      = 3.405e19
Chinchilla 70B:  6 × 70e9 × 1.4e12                  = 5.880e23
760M × 22GB:     6 × 760e6 × 22e9                   = 1.0032e20
1B × 22GB:       6 × 1e9 × 22e9                     = 1.3200e20
local time:      1.00e20 / 0.954e12                 = 1.05e8 s = 1,216 d
8×H100 time:     1.00e20 / (8×989e12×0.35)          = 3.61e4 s = 10.0 h
```
