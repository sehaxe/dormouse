# Post-training at ~1B on one 16 GiB card: what is measured, what transfers, what to run first

Scope: a 95M-parameter byte-level dense transformer (vocab 256, byte patching under test,
1M-context design), pretraining at BPB 1.52 on an English held-out window after 246 MB of
bytes, one RTX 5070 Ti (16 GiB). Question: what post-training actually buys capability at
this scale, ranked by measured delta per GPU-hour.

Tags: **[T]** = stated by the cited paper/repo (id + table/section); **[C]** = my arithmetic
on cited numbers; **[S]** = speculation / transfer not measured at the target scale;
**[M]** = measured in this repo. Russian prose, English tables.

---

## 0. TL;DR (RU)

1. **Посттренинг на 95M–1B покупает формат и извлечение уже имеющегося навыка, а не новое
   рассуждение.** Единственное измеренное исключение — дистилляция в 1.5B: R1-distill-1.5B
   даёт AIME24 28.9 против 10.0 у Qwen2.5-Math-1.5B-Instruct [T, 2501.12948 Table 5;
   T, 2409.12122 Table 5]. Это перенос распределения учителя в студента, у которого хватает
   ёмкости; при 95M такой ёмкости нет [S].
2. **RLVR на 95M — не «не доказано», а измеренный минус.** Единственный найденный *чистый
   RLVR*-прогон ниже 1B на GSM8K (GRPO поверх pretrained-модели, без SFT-обвязки):
   L20-Edu-135M, 1.82% → 1.59% (192 токена) и 1.21% (320 токенов), один прогон, автор сам
   называет это failure mode, а не нижней границей [T, 2606.22189 abstract]. На 0.6–1.5B
   RLVR работает, но только если база уже решает 10–50% задач, и стоит от ~100 A40-ч
   (Open-RS) до 11 500 A800-ч (JustRL) / 16 000 H100-ч (ProRL) [T, 2503.16219 Table 2;
   T, 2512.16649 §3; T, 2505.24864 §3.2].
3. **Самый дешёвый credible SFT:** 50k–1.1M примеров, 2 эпохи, `LR/BS ≈ 10e-6` (для 1.7B —
   9e-5 при batch 8), sum loss, маска на промпт, пакетирование с сохранением границ
   документов. На 95M полный SmolTalk (1.1M пар) — примерно **7 GPU-ч** при измеренных
   60 kB/s [C из M, `2026-11-07-architecture-frontier.md:242`; T, 2502.02737 Table 9].
   Узкое место — не обучение, а eval и подготовка данных.
4. **Оценка:** на 95M валидны только IFEval-подобный формат и held-out BPB. GSM8K, MBPP,
   BBH и MMLU у полу (1–2%), шум больше сигнала. На 1B добавить GSM8K и MBPP с 3 сидами:
   шум 1.0–1.6 пп на 0.6B MBPP за 10 прогонов [T, 2605.30478 Table 1], 0.9 пп на IFEval
   и 0.2–0.5 пп на MMLU/MATH/MBPP+ для 7B-класса [T, 2512.13961 §4.1.1].
5. **Байты:** опубликованного byte-level SFT нет. BLT не делал instruction tuning вообще —
   в тексте 0 упоминаний SFT и ни одного instruct-эксперимента [T, 2412.09871;
   проверено grep по HTML v1]. UTF8Tokenizer даёт протокол control-байтов и
   `apply_chat_template`, но без измерений SFT [T, 2510.16987 §3.1]. Главные поломки:
   нет атомарных special tokens, последовательности в 3.5–4.5 раза длиннее, формат-байты
   съедают лосс, один неверный байт делиметра ломает парсинг. Патчинг добавляет свою:
   граница prompt/response попадает внутрь патча [S].

**Честный потолок:** при 95M и BPB 1.52 модель не умеет арифметику на уровне, который
RLVR может усиливать. Посттренинг даст управляемый формат, короткие ответы и, возможно,
+2–5 пп на узкой задаче. Он не даст рассуждение. Если цель — «самый умный на FLOP»,
следующие GPU-часы лучше вложить в данные и дистилляцию, а не в RLVR.

---

## 1. Method and evidence rules

- Decomposed into five angles (capability per GPU-hour; SFT recipe; RLVR below 3B;
  evaluation/noise; byte-level post-training); 2+ sources per angle, primary arXiv HTML/PDF.
- Every number below is read from the paper's own table/text, not from an abstract claim
  where a table exists. Where only the abstract states a number, it says so.
- **Scale is stated on every row.** "Measured at 1.5B" and "measured at 7B" are never merged.
- Cost normalization is the weak point of the whole field: almost nobody reports GPU-hours.
  Where compute is reported, I use it; where it is not, the GPU-hour column is my estimate
  from the repo's measured 60 kB/s at 95M and is marked [C]. Cross-paper pp/GPU-h is **not**
  a metrology — it is an ordering of evidence.

---

## 2. Q1 — what moves capability at ~1B, ranked by measured delta per GPU-hour

### 2.1 Measured at ≤1.5B (the only rows that bear directly on a 1B design point)

| # | Intervention | Measured at | Baseline → result | Delta | Compute as published | Est. GPU-h at 95M [C] | Transfer to 95M |
|---|---|---|---|---|---|---|---|
| 1 | **SFT on curated data** (SmolTalk, 1.1M, 2 epochs) | 1.7B | GSM8K 31.1 → 47.54; IFEval 57.1 (base has no IFEval); MMLU-Pro 19.4 → 19.06 [T, 2502.02737 Tables 4, 10] | **+16.4 GSM8K; IFEval 57.1 absolute** | not reported | **~7 h** for 1.1M × 2 epochs | format yes; math no (capacity) [S] |
| 2 | **Rejection-sampled distillation / re-distillation** (1K samples from RL policy) | 1.5B | K&K 78.8 (SFT) → 82.0 (8k eval), beats DeepSeek-V3-0324; +2.4 after 25 RL steps [T, 2505.17988 §4] | **+3.2 K&K** | not reported | **~2–5 h** training; generation extra | highest delta/sample, needs teacher+verifier [S] |
| 3 | **Long-CoT distillation** (R1, 800k samples, SFT only) | 1.5B | AIME24 10.0 → 28.9; MATH-500 83.9 [T, 2501.12948 Table 5; T, 2409.12122 Table 5] | **+18.9 AIME24** | 800k samples, no GPU-h reported | **~40 h** training + teacher generation | capacity-bound; at 95M likely ≈0 [S] |
| 4 | **RLVR, math** (GRPO, 7k samples, 6 rollouts) | 1.5B | Avg 48.9 → 56.3; AIME24 28.8 → 46.7; AMC23 62.9 → 80.0 [T, 2503.16219 Table 1] | **+7.4 avg** | 4×A40 × 24 h = **96 A40-h**, $42 [T, Table 2] | **~100–200 h** (rollout-bound) | only if base ≥10–20% solve rate [C] |
| 5 | **RLVR, code** (LoRA, GRPO/GSPO, unit-test reward) | 0.6B / 1B | MBPP 0.273 → 0.417 (Qwen3-0.6B); +13.9…+14.4 pp [T, 2605.30478 Table 1] | **+14.4 pp** | not reported (10 runs) | **~10–30 h** | sign likely transfers, magnitude unknown [S] |
| 6 | **RLVR, search/tool** (GRPO, Wikipedia tool) | 0.8B | EM 0.092 → 0.352 (3.8×), 3 seeds [T, 2609.28765 §6.1] | **+26 pp EM** | not reported (9 runs × 310 steps) | **~50–150 h** (rollout-bound) | tool schema is the hard part, not the model [S] |
| 7 | **DPO** (Tulu 3 preference mix) | 8B only | Avg 60.4 → 64.4; GSM8K +8.1; MATH +10.5; IFEval +8.3 [T, 2411.15124 Table 4] | **+4.0 avg** | not reported | **~2–5 h** at 95M | no ≤1B ablation found; likely style only [S] |
| 8 | **Tool-use traces SFT** | 30B only | SWE-V 59.7 → 63.5 with traces vs 60.5 without [T, 2610.22019 Table 2] | **+3.8 SWE-V** | 500B tokens midtraining [T, Table 1] | **10^3–10^4 h** | no ≤1B measurement [S] |
| 9 | **ORPO** | 7–8B only | no ≤1B number found in this search | — | — | — | unmeasured [S] |

### 2.2 Measured at 7–8B (contrast; do not read as 1B numbers)

| Intervention | Measured at | Baseline → result | Delta | Source |
|---|---|---|---|---|
| SFT → DPO → RLVR, full Tulu 3 | 8B | Avg 60.4 → 64.4 → 64.8; GSM8K 76.2 → 84.3 → 87.6; MATH 31.5 → 42.0 → 43.7; IFEval 72.8 → 81.1 → 82.4 | DPO **+4.0**, RLVR **+0.4** | [T, 2411.15124 Table 4] |
| SFT → DPO → RLVR, Olmo 3 Think | 7B | AIME25 57.6 → 62.7 → 64.6; MATH 94.4 → 92.4 → 95.1; IFEval 77.9 → 75.9 → 88.2; IFBench 30.0 → 28.3 → 41.6 | DPO **+5.1 AIME25**, RLVR **+1.9 AIME25 / +12.3 IFEval** | [T, 2512.13961 Table 15] |
| ProRL (prolonged RL, 2k+ steps) | 1.5B | +14.7 math, +13.9 coding, +54.8 logic, +25.1 STEM, +18.1 IF | **16 000 H100-h** | [T, 2505.24864 §3.2] |
| JustRL (single-stage GRPO, fixed HPs) | 1.5B | 54.9 / 64.3 avg on 9 math benchmarks; 4000+ steps | **32×A800 × 15 days ≈ 11 500 A800-h** | [T, 2512.16649 §3.1] |

### 2.3 Ranking (blunt)

**By measured delta per GPU-hour at ≤1.5B:** SFT (format/task elicitation) > re-distillation >
code RLVR > search RLVR > math RLVR > long-CoT distillation > DPO (no 1B data) > tool traces
(no 1B data) > ORPO (no 1B data).

Only five interventions publish enough to compute a rate at all. The two orders of magnitude
between them are the whole story:

| Rank | Intervention | Best measured delta | Published compute | Rate (pp per 100 GPU-h) |
|---|---|---|---|---|
| 1 | **Open-RS RLVR** (1.5B math) | +7.4 avg [T, 2503.16219 Table 1] | 96 A40-h [T, Table 2] | **7.7 pp / 100 A40-h** |
| 2 | **JustRL** (1.5B math) | 54.9 / 64.3 avg; baseline not stated in the abstract | ~11 500 A800-h [T, 2512.16649 §3.1] | ~0.1 pp / 100 A800-h [C] |
| 3 | **ProRL** (1.5B, boundary-expanding) | +14.7 math [T, 2505.24864 abstract] | 16 000 H100-h [T, §3.2] | **0.09 pp / 100 H100-h** |
| 4 | SFT, re-distill, code/search RLVR, DPO | +3.2…+16.4 pp | **not reported** | cannot compute |
| 5 | Tool traces (30B), ORPO (7–8B) | +3.8 SWE-V at 30B | not reported | cannot compute; no ≤1B data |

The rate gap is not a GPU-speed artifact: A40 and H100 are the same order of magnitude. It is
the difference between *sharpening* a behavior the base already has (Open-RS, 96 h) and
*expanding* the boundary (ProRL, 16 000 h). At 95M, only the first is affordable — and the
first is exactly the one that needs a base which already solves 10–50% of the task.

Two corrections to that ordering:

- **Long-CoT distillation has the largest absolute delta (+18.9 AIME24) but the worst
  capacity transfer.** The student is 1.5B and already math-specialized; 800k R1 traces
  teach it a *format* it can execute. At 95M the same traces cannot be executed, so the
  delta collapses toward zero. Do not budget for it at 95M [S].
- **ProRL/JustRL prove RLVR can expand boundaries at 1.5B, and simultaneously prove the
  price is 10^4 GPU-hours.** On one 5070 Ti that is years, not hours. The affordable RLVR
  is the 96-A40-hour Open-RS recipe — and it starts from an R1-distilled base that already
  has the long-CoT behavior [T, 2503.16219 Table 1].

**The one number that kills the naive plan:** a 95M byte model's GSM8K is ~1–2% (the closest
measured point: 1.82% at 134.5M [T, 2606.22189 abstract]). With pass@1 = 1%, a group of 8
rollouts is all-zero on 92% of prompts (`0.99^8 = 0.923` [C]); with 5%, 66%; with 20%, 17%.
RLVR spends most of its compute on prompts that produce no gradient. This is the same
`p(1−p)` starvation argument as [T, 2610.34894 App. B.2], now with a measured 134.5M point.

---

## 3. Q2 — the cheapest credible SFT recipe on one card

### 3.1 Open recipes, with numbers

| Recipe | Model | Data | Epochs / schedule | Result | Source |
|---|---|---|---|---|---|
| **Tulu 3 SFT** | 8B / 70B | 939,344 examples (Table 6); 425,145 preference prompts | 2 epochs, LR 5e-6, **sum loss**, no token-mean | 8B SFT avg 60.4; 70B 72.6 | [T, 2411.15124 §4.1, Table 6, Table 13] |
| **SmolLM2 SFT** | 1.7B | SmolTalk 1.1M pairs (MagPie-Ultra 431k, OpenHermes 100k, NuminaMath 112k, …) | 2 epochs, seq 8192, batch 128, LR 3e-4 | IFEval 57.1, GSM8K 47.54, MATH 19.64, MMLU-Pro 19.06 | [T, 2502.02737 Table 9, Table 10] |
| **SmolLM2 DPO** | 1.7B | UltraFeedback 2 epochs | LR 1e-6, beta 0.5, batch 128, seq 1024 | used as final alignment stage; no ablation | [T, 2502.02737 §5.3] |
| **SmolTulu** (full pipeline, DPO-1130) | 1.7B | Tulu 3 SFT mixture, contamination-checked (GSM8K 0.0%, IFEval 0.0%) | **LR 9e-5, batch 8 → LR/BS 11.25e-6** vs Tulu 3 8B 0.039e-6 | IFEval 67.7 (+11%), GSM8K 51.6 (+3.4), ARC 57.1 (+5.4) | [T, 2412.08347 Table 1, Table 2, abstract] |
| **LIMA** | 65B | 1,000 curated prompts/responses | standard SFT | strong format following, no RL | [T, 2305.11206 abstract] |
| **LIMO** | 32B | 817 examples | SFT | AIME24 63.3, MATH500 95.6 | [T, 2502.03387 abstract] |
| **OpenThoughts3** | 7B | 1.2M examples, QwQ-32B teacher | SFT | AIME25 53, LCB 51, GPQA 54 | [T, 2506.04178 abstract] |
| **R1-distill** | 1.5B | 800k R1 samples | SFT only, no RL | AIME24 28.9, MATH-500 83.9 | [T, 2501.12948 §2.3.3, Table 5] |
| **Zephyr** | 7B | UltraFeedback 200k (AIF) | dSFT then dDPO | MT-Bench SOTA for 7B at release, beats Llama2-Chat-70B | [T, 2310.16944 abstract] |

### 3.2 What the numbers actually say about data size

- **At 1.7B, two epochs beat one on the same mix:** SmolTalk 1 epoch → SmolLM2-SFT 2 epochs:
  IFEval 46.67 → 57.09 (+10.4), GSM8K 43.75 → 47.54 (+3.8), MATH 18.60 → 19.64 [T, 2502.02737
  Table 10]. This is the cleanest "more epochs help at small scale" measurement I found.
- **Data choice beats hyperparameters by a wide margin at 1.7B:** one epoch on OpenHermes
  gives GSM8K 42.91 but IFEval 30.01; one epoch on MagPie-Ultra+ gives IFEval 48.16 but
  GSM8K 19.94; SmolTalk gives 43.75 / 46.67 [T, 2502.02737 Table 10]. A 28-point GSM8K
  swing from data alone. At this scale, selection is the lever, not LR.
- **At 8B, the full 939k mix beats every subsample** (5%, 10%, 25%, 50%, 75%, 100%):
  "our full mix performs best overall" [T, 2411.15124 §4.1, Fig. 4]. There is no published
  "1k examples is enough at 1B" result. LIMA/LIMO are 32–65B; do not transfer them down [S].
- **Capacity-dependent selection:** with 1.5B–8B students and stronger teachers, high-likelihood
  responses give faster early gains for **smaller** models, while low-likelihood data wins for
  larger models on longer training [T, 2608.13721 abstract]. At 95M, prefer responses close to
  the student's distribution (filter by student likelihood), not the hardest teacher traces.
- **Rejection sampling is the cheapest way to make SFT data better:** re-distilling 1K samples
  from the RL-trained policy reached 82.0 on K&K vs 78.8 for the long-CoT SFT baseline, and
  beat DeepSeek-V3-0324 without RL [T, 2505.17988 §4]. The filter matters more than the volume.

### 3.3 Concrete recipe for this repo

| Knob | Value | Why |
|---|---|---|
| Examples | **50k–200k** for format; **1.1M** if you want the full SmolTalk mix | 1.1M × 2 epochs at 60 kB/s ≈ 7 h; the model is cheap enough that volume is not the constraint [C] |
| Selection | dedup (MinHash/LSH), benchmark decontamination (13-gram / 0.6 LCS), quality filter, then **student-likelihood filter** (keep middle band) | Tulu 3 decontamination; SmolLM2's 13-gram rule; 2608.13721 capacity result [T] |
| Packing | pack to fixed byte budget; **never cross document boundaries without a mask**; keep the prompt mask exact | Tulu 3 loss-aggregation warning; Olmo 3 document packing + intra-document masking [T, 2411.15124 §4.3.2; T, 2512.13961 §3.6.4] |
| Loss | **sum loss / token-weighted**, not per-example mean | Tulu 3: gradient accumulation with per-example mean silently changes sample weighting [T, 2411.15124 §4.3.2] |
| LR | `LR/BS ≈ 10e-6` as the starting ratio (SmolTulu 11.25e-6 at 1.7B); sweep 3 values | SmolTulu Table 2; Tulu 3's 5e-6 at 8B is 0.039e-6 — 250× lower ratio [T] |
| Schedule | warmup 5–10%, constant/cooldown to 0.1× peak, 2 epochs | Tulu 3 2 epochs; SmolLM2 2 epochs [T] |
| Sequence | 2k–8k **bytes** (not 1M); patch size 8 → 256–1024 positions | SFT conversations are short; long context is a pretraining property [S] |
| Seeds | **3 seeds minimum**, report SD | Tulu 3 8B seed range 0.3 avg; 70B range 2.6 [T, 2411.15124 Table 13] |

### 3.4 Failure modes (measured, not folklore)

| Failure | Evidence | What it looks like here |
|---|---|---|
| **Catastrophic forgetting of BPB** | SmolLM2 base ARC 60.5 → instruct 51.7 (−8.8; protocols may differ, treat as direction), MMLU-Pro 19.4 → 19.3 (flat) [T, 2502.02737 Tables 4, 5] | held-out BPB rises while IFEval rises. Always report both; a +5 IFEval with +0.15 BPB is a regression on the stated goal |
| **Format collapse / delimiter leakage** | Olmo 3: including `<|im_start|>`-style tokens in midtraining makes the model emit them at inference; GSM8K 49.43 → **0**, CruxEval 32.89 → 18.91; plain-text template 46.02 / 29.65 [T, 2512.13961 §3.5.4] | at byte level there are no atomic tokens; the analogue is the model repeating the literal delimiter bytes or never closing the turn |
| **Length drift / language drift** | Open-RS: after 150–200 steps, multilingual output re-emerges and accuracy degrades below baseline; completion cap 3584–4096 truncates hard problems [T, 2503.16219 §5, Limitations] | byte model drifts into repeating the prompt or switching language; cap the response and measure `fin@1` |
| **Length penalty collapses exploration** | JustRL: explicit length penalties and "robust" verifiers can degrade performance [T, 2512.16649 abstract] | do not add a length reward until the base format is stable |
| **Seed variance** | Tulu 3 8B: 59.9/60.1/59.8/59.8/59.8; 70B: 71.8/70.0/72.6 [T, 2411.15124 Table 13] | a 1-run "win" at 95M is not evidence; 3 seeds or nothing |
| **Gradient-accumulation weighting** | Tulu 3: per-example mean vs token mean changes results [T, 2411.15124 §4.3.2] | mandatory on a 16 GiB card; check the loss reduction, not just the optimizer |

---

## 4. Q3 — RLVR at 1B: does it work?

### 4.1 Evidence, both directions

| Study | Model size | Task / reward | Result | Verdict |
|---|---|---|---|---|
| **Open-RS** [T, 2503.16219] | **1.5B** (R1-distill) | MATH, accuracy + format, 7k samples, 6 rollouts | Avg 48.9 → 56.3; AIME24 28.8 → 46.7; AMC23 62.9 → 80.0; 96 A40-h | **works, cheap, but base is already long-CoT** |
| **ProRL** [T, 2505.24864] | **1.5B** | math/code/STEM/logic, 2k+ steps, 16k H100-h | +14.7 math, +13.9 coding, +54.8 logic; pass@k expansion | **works, costs 10^4 GPU-h** |
| **JustRL** [T, 2512.16649] | **1.5B** | 9 math benchmarks, single-stage GRPO | 54.9 / 64.3 avg; 4000+ steps; ~11.5k A800-h | **works, still 10^4 GPU-h** |
| **Code RLVR** [T, 2605.30478] | **0.6B / 1B** | MBPP, unit tests, LoRA | 0.273 → 0.417 (+14.4 pp), 10 runs, p=5e-6; Ruff-only **hurts** (−2.0 pp) | **works below 1B, narrow task** |
| **Search RLVR** [T, 2609.28765] | **0.8B** | MuSiQue, EM/F1 + format, 3 seeds | 0.092 → 0.352 EM (3.8×); EM-only worst at all seeds | **works below 1B, tool schema-dependent** |
| **SFT-then-RL SLM** [T, 2606.04466] | **0.5B / 1B** | GSM8K, stage-specific data | Avg 42.16 → 45.27 (0.5B), 39.21 → 43.03 (1B); full pipeline 48.53 | **works, but SFT+data recycling is most of it** |
| **Pencil probe** [T, 2611.02574] | **1B** (OLMo-2) | GSM8K, 13 checkpoints | First-try 58.0 → 61.3; pass@256 94.5 → 96.9; error probes unchanged | **+3.3 pp, no new coverage — sharpening only** |
| **L20-Edu-135M** [T, 2606.22189] | **134.5M** | GSM8K, GRPO | 1.82% → 1.59% (192 tok), 1.21% (320 tok); single run | **measured negative at ~100M** |
| **Yue et al.** [T, 2504.13837] | 7B-class | pass@k large k | RLVR models do not exceed base coverage | **no new capability at 7B either** |

### 4.2 The answer

**Below ~3B, RLVR works as a *sharpener*, not as a teacher.** The mechanism is consistent
across every measured case: the base already contains the solution path with some probability,
and RLVR raises the probability of selecting it. The 1B probe is the cleanest demonstration:
first-try GSM8K +3.3 pp, but pass@256 coverage only 94.5 → 96.9, and the internal
error-detection probe is invariant across all 13 checkpoints [T, 2611.02574 abstract].
The model does not learn a better notion of "wrong"; it learns to answer sooner.

**At 95M the sharpener has nothing to sharpen.** GSM8K is 1–2%; a group of 8 is all-zero on
>92% of prompts [C]. The one measured 134.5M attempt went *backwards* [T, 2606.22189].
So the honest answer to "does RLVR work at 1B?" is:

- **At 1–1.5B, yes, on a base that already solves 10–50% of the task, with 10^2–10^3 GPU-h
  for a modest recipe and 10^4 GPU-h for a boundary-expanding one.**
- **At 95M–135M, no measurable gain; one measured loss.** Do not run RLVR as the first
  post-training experiment.

### 4.3 Minimum rollout budget and reward shape (if you run it anyway)

| Item | Minimum credible value | Evidence |
|---|---|---|
| Prompts | 500–1,000 in the **1–6 of 8** solve band; if <100 survive, abort | P1 `p(1−p)` starvation [T, 2610.34894 App. B.2]; Open-RS uses 7k samples [T, 2503.16219] |
| Rollouts / prompt | 6–8 | Open-RS 6 [T, Table 2]; ProRL 16 [T, §3.2] |
| Response cap | 3.5k–4k tokens; longer truncates hard problems | Open-RS 3584/4096 [T, Limitations] |
| Steps | 50–200, not 500+; degradation starts after ~200 | Open-RS drops below baseline after 200 steps [T, §5] |
| Reward | **unit-test / exact-match + format**; do **not** use a static-analysis-only or length-only proxy | Code RLVR: Ruff-only −2.0 pp; combined +14.4 pp [T, 2605.30478 Table 1]; Open-RS: EM-only worst at every seed [T, 2609.28765 Table 1] |
| LR | 1e-6 constant, no KL, no entropy bonus (JustRL) | [T, 2512.16649 Table 2] |

### 4.4 Known collapse modes

1. **All-zero groups** — no gradient; the dominant failure at 95M [C from binomial].
2. **Entropy collapse / length collapse** — ProRL needs KL control and reference resets to run
   2k+ steps [T, 2505.24864 §2.2.1]; JustRL warns that length penalties and "robust" verifiers
   can collapse exploration [T, 2512.16649 abstract].
3. **Language/format drift** — Open-RS: multilingual output returns after 150–200 steps
   [T, 2503.16219 §5].
4. **Proxy reward hacking** — static-analysis reward shortens completions without improving
   correctness; the paper calls it "systematic behavioral shift" [T, 2605.30478 abstract].
5. **Benchmark overfitting** — Olmo 3's RL gains on IFEval/IFBench are large (+12.3/+13.3 at
   7B) while AIME24 moves −3.0; the reward selects the benchmark you wrote, not general
   reasoning [T, 2512.13961 Table 15].

---

## 5. Q4 — evaluation that is not BPB

### 5.1 What is valid at 95M vs 1B

| Benchmark | 95M | 1B | Noise / seed variance | Source |
|---|---|---|---|---|
| **Held-out BPB** (English window) | **valid, primary** | valid, primary | repo paired-eval resolution 0.002–0.005 BPB [M, `.bulba`/AGENTS.md §3.1] | [M] |
| **IFEval** (loose/strict) | **valid; the only capability metric above floor** | valid | SD 0.88 at 7B (3 runs × 14 models); high-variance bucket | [T, 2512.13961 §4.1.1] |
| **GSM8K** | **at floor (~1–2%)** | valid if base ≥5% | SD 0.54 (AIME24) at 7B; 1.0–1.6 pp at 0.6B MBPP over 10 runs | [T, 2512.13961; T, 2605.30478 Table 1] |
| **MBPP / MBPP+** | at floor | valid if base ≥5% | SD 0.27 (MBPP+) at 7B; 1.0–1.6 pp at 0.6B | [T, 2512.13961; T, 2605.30478] |
| **HumanEval / HumanEval+** | at floor | valid | SD 0.46 at 7B | [T, 2512.13961] |
| **BBH** | at floor | weak signal | SD 0.39 at 7B | [T, 2512.13961] |
| **MMLU / MMLU-Pro** | at chance (25%) | weak; MMLU-Pro floor ~19 at 1.7B | SD 0.22 (MMLU) at 7B | [T, 2512.13961; T, 2502.02737 Table 5] |
| **AIME24/25** | meaningless | high variance; 30 problems | SD 0.54 at 7B | [T, 2512.13961] |
| **GPQA** | meaningless | high variance | SD 1.48 at 7B — highest in the suite | [T, 2512.13961] |
| **AlpacaEval 2** | meaningless | style, not capability | SD 1.24 at 7B | [T, 2512.13961] |

### 5.2 The smallest set that predicts usefulness

**At 95M:** (1) held-out BPB; (2) IFEval loose + strict; (3) a 200-item custom format suite
(JSON validity, delimiter closure, answer-in-box, refusal). That is the whole set. GSM8K/MBPP/
BBH/MMLU add noise, not signal, at 1–2% accuracy.

**At 1B:** add GSM8K (3 seeds, 1319 items) and MBPP (3 seeds, 500 items). Only interpret a
difference if it exceeds **2× the seed SD**; at 0.6B that is ~2–3 pp on MBPP [C from
T, 2605.30478 Table 1]. Use `pass@1` **and** `pass@8`/`pass@256`; the pencil result shows
first-try and coverage diverge [T, 2611.02574].

### 5.3 Statistical floor

- Standard error of a benchmark with `n` items at accuracy `p`: `sqrt(p(1−p)/n)` [T, 2411.00640
  §2]. GSM8K n=1319 at p=0.5 → 1.38 pp; MMLU n=14k → 0.42 pp; a 200-item format suite at
  p=0.8 → 2.8 pp [C]. **A 200-item suite cannot resolve a 2 pp difference.**
- TinyBenchmarks: 100 curated MMLU examples reproduce the full MMLU ranking [T, 2402.14992
  abstract] — useful for cheap *ranking*, not for absolute capability, and not at 95M where
  MMLU is chance.
- Olmo 3's variance buckets are the best published per-benchmark noise table for a 7B-class
  model: high variance GPQA 1.48, AlpacaEval 1.24, IFEval 0.88; stable ZebraLogic 0.56,
  AIME24 0.54, HumanEval+ 0.46, BBH 0.39; very stable LCB 0.29, MBPP+ 0.27, MATH 0.25,
  MMLU 0.22, PopQA 0.16 [T, 2512.13961 §4.1.1]. **Do not import these as 1B numbers** —
  variance is scale-dependent and the 1B model is closer to the floor, so its binomial
  variance is larger for the same n.
- Tulu 3 seed spread: 8B average 59.9–60.1 (range 0.3), 70B 70.0–72.6 (range 2.6)
  [T, 2411.15124 Table 13]. At 1B, expect the 70B pattern, not the 8B one [S].

---

## 6. Q5 — byte-level specifics

### 6.1 What breaks

| Issue | At token level | At byte level | Evidence |
|---|---|---|---|
| **Loss mask alignment** | mask on token ids; BPE boundary can split a word | mask on **byte offsets** — actually *more* precise, no BPE boundary problem | [C from architecture] |
| **Patching boundary** | n/a | a patch of 8 bytes can contain the prompt/response boundary; the patch's loss mixes both | [S] — no published measurement |
| **Special/format tokens** | `<|im_start|>` is one id; atomic | vocab 256 has no free ids; delimiter is a multi-byte string; one wrong byte breaks the parse | [T, 2510.16987 §3.1] (C0 control bytes); [T, 2512.13961 §3.5.4] (special-token leakage at token level) |
| **Sequence length** | 1 token ≈ 4 chars | 1 byte = 1 position; same text is 3.5–4.5× longer | [T, 2412.09871 Table 1] bytes/patch 4.4–6.1; [C] |
| **Format bytes in the loss** | space/newline are merged into tokens | every space/newline is a prediction; formatting is a large share of SFT loss | [S] — architectural inference |
| **Chat template** | tokenizer's `apply_chat_template` | must be a byte protocol (C0 control bytes) or a literal string | [T, 2510.16987 §3.1, Table 1] |
| **BPB vs accuracy** | separate metrics | BPB can *rise* while IFEval rises; the two are not monotonically related | [C] from SmolLM2 base/instruct ARC–BPB divergence [T, 2502.02737] |

**The patching boundary is the one that will bite this repo first.** With 8-byte patches,
a 4k-byte SFT example is 512 positions; a prompt of 3,999 bytes and a response starting at
byte 4,000 puts the boundary in the middle of patch 500. If you train on the whole patch,
you train on prompt bytes; if you mask the patch, you lose the first 1–7 response bytes.
The fix is boring: pad the prompt to a patch boundary (or insert a 1–7 byte filler that is
masked) so every patch is entirely prompt or entirely response. No paper measures the cost
of not doing this [S].

### 6.2 Published byte-level work and what it did for instruction data

| Work | Model | Post-training | Instruction data | Source |
|---|---|---|---|---|
| **BLT** (Byte Latent Transformer) | 400M–8B, byte patches | **none** — pretraining + zero/few-shot eval only | **none.** The paper has no SFT section; grep of the HTML v1 finds 0 "SFT", 1 "Instructions" (page boilerplate), 1 "fine-tuned" (related work). Evaluation is ARC/HellaSwag/PIQA/MMLU/MBPP/HumanEval with prompt scoring [T, Table 1] | [T, 2412.09871] |
| **UTF8Tokenizer** | tokenizer only | instruction tuning supported via `apply_chat_template`; **no measured SFT results** | C0 control bytes repurposed: message blocks, tool calls, thinking spans; 256 IDs exactly, no id 256 | [T, 2510.16987 §3.1, Table 1] |
| **Broken Tokens?** | 20 benchmarks, instruction-tuned models | none (robustness study) | shows instruction-tuned models retain up to 93.4% under random tokenization and 90.8% under char-level; char-level improves string manipulation/code by up to +14% | [T, 2506.19004 abstract] |
| **ByT5 / bGPT / MegaByte** | byte/char models | none | no instruction tuning; pretraining/classification only | [T, 2105.13626; T, 2402.19155; T, 2305.07185] |

**There is no published byte-level SFT recipe.** That is the honest state of the art as of
2026-11-08. The closest thing to a protocol is UTF8Tokenizer's control-byte table, and it
has no SFT numbers. Anything this repo does for byte-level SFT is a first measurement, not
a replication.

### 6.3 What to do here

1. **Do not add a 257th token.** Keep vocab 256; encode roles with C0 control bytes
   (UTF8Tokenizer's table) or a fixed literal delimiter. Adding an id breaks the embedding
   table's 256×d shape and the "no tokenizer" claim.
2. **Mask by byte offset, then aggregate to patches.** If a patch straddles the boundary,
   pad the prompt to a patch boundary before tokenizing. This is a data-loader rule, not a
   model change.
3. **Measure format bytes separately.** Report `loss_on_format_bytes / loss_on_content_bytes`.
   If format dominates, the model is learning delimiters, not answers.
4. **Use 2k–8k byte sequences, not 1M.** The 1M design is a pretraining property; SFT
   conversations are short. At 8 bytes/patch that is 256–1024 positions — cheap.
5. **Prefer IFEval + a custom byte-format suite over GSM8K.** At 95M GSM8K is at floor; the
   format suite is the only thing that can move.

---

## 7. VERIFIED (sourced) vs SPECULATION (my inference)

### VERIFIED

- SFT at 1.7B: SmolTalk 1.1M × 2 epochs, seq 8192, batch 128, LR 3e-4 → IFEval 57.09,
  GSM8K 47.54, MATH 19.64; 1 epoch → IFEval 46.67, GSM8K 43.75 [T, 2502.02737 Table 10].
- SFT at 8B: 939,344 examples, 2 epochs, LR 5e-6, sum loss; full mix beats 5–75% subsamples
  [T, 2411.15124 §4.1, Table 6, Fig. 4].
- SmolTulu LR/BS 11.25e-6 at 1.7B vs Tulu 3 8B 0.039e-6; IFEval 67.7, GSM8K 51.6
  [T, 2412.08347 Table 2, abstract].
- Tulu 3 8B stage deltas: SFT 60.4 → DPO 64.4 → RLVR 64.8; GSM8K 76.2 → 84.3 → 87.6;
  MATH 31.5 → 42.0 → 43.7 [T, 2411.15124 Table 4].
- Olmo 3 7B Think stage deltas: AIME25 57.6 → 62.7 → 64.6; IFEval 77.9 → 75.9 → 88.2;
  IFBench 30.0 → 28.3 → 41.6 [T, 2512.13961 Table 15].
- R1-distill-1.5B: AIME24 28.9, MATH-500 83.9, GPQA 33.8, LiveCodeBench 16.9
  [T, 2501.12948 Table 5]; Qwen2.5-Math-1.5B-Instruct AIME24 3/30 = 10.0
  [T, 2409.12122 Table 5].
- Open-RS 1.5B: 7k samples, 6 rollouts, 4×A40 24 h, $42; avg 48.9 → 56.3; AIME24
  28.8 → 46.7; AMC23 62.9 → 80.0; degradation after 200 steps [T, 2503.16219 Tables 1–2, §5].
- ProRL 1.5B: +14.7 math / +13.9 coding / +54.8 logic; 16k H100-h
  [T, 2505.24864 §3.2].
- JustRL 1.5B: 32×A800 × ~15 days; 54.9/64.3 avg; 4000+ steps; length penalties can hurt
  [T, 2512.16649 §3.1, abstract].
- Code RLVR: Qwen3-0.6B MBPP 0.273 → 0.417 (+14.4 pp), Llama3.2-1B similar; Ruff-only
  −2.0 pp; 10 runs, SD 0.010–0.016 [T, 2605.30478 Table 1].
- Search RLVR: Qwen3.5-0.8B, 0.092 → 0.352 EM (3.8×), 3 seeds; EM-only worst at all seeds
  [T, 2609.28765 §6.1, Table 1].
- OLMo-2-1B pencil: GSM8K first-try 58.0 → 61.3, pass@256 94.5 → 96.9; probes invariant
  [T, 2611.02574 abstract].
- L20-Edu-135M: GSM8K 1.82% → 1.59% (192 tok), 1.21% (320 tok), single run
  [T, 2606.22189 abstract].
- Olmo 3 variance: GPQA 1.48, AlpacaEval 1.24, IFEval 0.88; AIME24 0.54, HumanEval+ 0.46,
  BBH 0.39; LCB 0.29, MBPP+ 0.27, MATH 0.25, MMLU 0.22 [T, 2512.13961 §4.1.1].
- Tulu 3 seed spread: 8B 59.9–60.1; 70B 70.0–72.6 [T, 2411.15124 Table 13].
- Olmo 3 special-token leakage: GSM8K 49.43 → 0; CruxEval 32.89 → 18.91; plain-text
  template 46.02 / 29.65 [T, 2512.13961 §3.5.4].
- BLT: 8B FLOP-matched avg Llama 3 60.0, BLT-Space 58.0, BLT-Entropy 61.1; bytes/patch
  4.4/6.1/4.5; **no SFT/instruct experiments** [T, 2412.09871 Table 1; grep].
- UTF8Tokenizer: 256 IDs, C0 control bytes, `apply_chat_template`, 14× faster tokenization,
  8× less host-device transfer; no SFT results [T, 2510.16987 §3.1].
- Broken Tokens: up to 93.4% retained under random tokenization, 90.8% char-level; char-level
  +14% on string manipulation/code [T, 2506.19004 abstract].
- Repo throughput: ~60 kB/s at 95M; 1 GB ≈ 4.6 GPU-h [M, `2026-11-07-architecture-frontier.md:242`].
- Repo pretraining: 95,035,200 params, 12 layers, BPB 1.519 at 246 MB [M, `.bulba/goal.md`;
  M, `2026-11-08-efficiency-frontier.md:156`].

### SPECULATION (my inference, explicitly marked)

- **S1. At 95M, SFT on SmolTalk will move IFEval but not GSM8K.** The capacity argument:
  a 134.5M model scores 1.82% on GSM8K before RLVR [T, 2606.22189], so there is almost no
  math behavior to elicit. Prediction: +15–30 pp IFEval, +0–2 pp GSM8K. Not measured.
- **S2. Distillation at 95M will transfer format, not reasoning.** R1-distill-1.5B's +18.9
  AIME24 is a 1.5B student executing a 671B teacher's trace format [T, 2501.12948]. A 95M
  student has ~16× less capacity and a byte-level output; the trace format may transfer,
  the arithmetic will not. Prediction: +5–10 pp on a narrow arithmetic format, ≈0 on AIME.
- **S3. The patch-boundary mask will cost 0.5–2 pp on a 4k-byte SFT task** if left unmasked,
  because ~1/512 of positions are contaminated and the delimiter sits exactly there. Not
  measured by anyone.
- **S4. A 200-item format suite has a 2.8 pp standard error at p=0.8** [C from 2411.00640];
  a "3 pp win" on it is noise. Use 500+ items or 3 seeds.
- **S5. The 60 kB/s figure will drop 2–4× under SFT** because of variable-length packing,
  attention at 8k, and eval interleaving. Costs in §8 carry a 2× contingency.
- **S6. DPO at 95M will buy style and refusal behavior, not reasoning.** DPO's +4.0 avg at
  8B [T, 2411.15124 Table 4] is a preference shift on a model that already reasons; at 95M
  the preference signal is mostly length/format. No ≤1B DPO ablation exists.

---

## 8. Ranked, costed plan: first three post-training experiments

Cost basis: 60 kB/s at 95M [M], i.e. 216 MB/h. SFT bytes = examples × average bytes × epochs.
Generation is costed separately at 0.3–1 kB/s (byte-level, batch 1, 0.5–1.5B teacher) [C].
All costs include 2× contingency (S5) and eval.

### E1 — Format SFT (the only experiment with a guaranteed measurable signal)

| Field | Value |
|---|---|
| Data | 20k examples: 8k IFEval-style constraint prompts, 4k JSON/structured output, 4k short QA, 4k refusal/format; decontaminated against the eval set |
| Training | 2 epochs, seq 4k bytes, batch 32 (grad accumulation), LR/BS ≈ 10e-6, sum loss, prompt mask, patch-aligned boundaries |
| Cost | 20k × ~1 kB × 2 = 40 MB → **0.2 h** training; **6–10 GPU-h** with 3 seeds and eval |
| Pre-registered pass | IFEval loose **≥ +25 pp absolute** over the pretrain checkpoint; held-out BPB **≤ +0.03**; 3/3 seeds same sign |
| Pre-registered fail | < +10 pp, or BPB > +0.10, or 2/3 seeds negative → stop and fix data/format before anything else |
| Why first | It is the only intervention whose mechanism (format compliance) is capacity-independent; SmolLM2 1.7B gains +10.4 IFEval from a second epoch alone [T, 2502.02737 Table 10] |

### E2 — Rejection-sampled distillation on one narrow verifiable task

| Field | Value |
|---|---|
| Teacher | 0.5–1.5B instruct model with a verifier (arithmetic or Python), not a 70B API |
| Data | 5k traces, keep only verified-correct; filter repetition loops (DEFLATE ratio < 0.05 over the last 20k chars [T, 2610.22236 §3.4]); student-likelihood middle band [T, 2608.13721] |
| Training | 1–2 epochs, seq 2k bytes, same LR rule as E1; start from the E1 checkpoint |
| Cost | generation 5k × 2 kB = 10 MB at 0.3–1 kB/s → **3–9 h**; training 10 MB × 2 → 0.1 h; **15–40 GPU-h** with 3 seeds and eval |
| Pre-registered pass | Task pass@1 **≥ +5 pp** over E1 at the same format; BPB **≤ +0.05**; 2/3 seeds positive |
| Pre-registered fail | < +2 pp, or the verifier accepts < 60% of traces (teacher too weak), or BPB > +0.10 |
| Why second | Re-distillation is the highest delta per sample at 1.5B: 1K samples, 78.8 → 82.0 [T, 2505.17988 §4]; it is also the only path that can add a skill the base lacks |

### E3 — RLVR on the E2 checkpoint (conditional; pre-registered abort)

| Field | Value |
|---|---|
| Precondition | Sample E2 8× on 500 prompts; keep only 1–6 of 8. **If <100 prompts survive, abort** — this is the measured starvation regime [C from T, 2610.34894 App. B.2; T, 2606.22189] |
| Training | GRPO, 6–8 rollouts, response cap 2k bytes, LR 1e-6 constant, no KL, no length penalty; 50–100 steps |
| Cost | 500 prompts × 8 × 600 B = 2.4 MB per iteration at 0.3–1 kB/s → 0.7–2.2 h; 50–100 iterations → **35–220 GPU-h**; budget **60 h** |
| Pre-registered pass | pass@1 **≥ +3 pp** over E2 with non-overlapping 95% CI across 3 seeds; format violation **≤ 1%**; no language drift |
| Pre-registered fail | no solve-rate band, or pass@1 ≤ E2, or format violation > 5% → abort and spend the 60 h on more distillation data |
| Why third, not first | At 95M the measured attempt went 1.82 → 1.59 [T, 2606.22189]; at 1.5B the cheap recipe costs 96 A40-h and needs an R1-distilled base [T, 2503.16219]; the expensive recipe costs 10^4 GPU-h [T, 2505.24864; T, 2512.16649] |

**Total for E1+E2: 21–50 GPU-h.** E3 is conditional and should be expected to fail at 95M.
If the goal is capability and not format, skip E3 and spend the 60 h on 3× more distillation
data and a second task.

---

## 9. Sources

Primary (arXiv HTML/PDF read; table/section cited inline):

1. 2411.15124v5 — Lambert et al., *Tulu 3*, 2024-11-22. SFT 939,344 / 2 epochs / LR 5e-6 /
   sum loss; Table 4 stage deltas; Table 13 seed variance; §4.3.2 loss aggregation.
2. 2502.02737v1 — Allal et al., *SmolLM2*, 2025-02-04. SmolTalk 1.1M / 2 epochs / seq 8192 /
   batch 128 / LR 3e-4; Tables 4, 5, 9, 10.
3. 2412.08347v1 — Alrashed, *SmolTulu*, 2024-12-11. LR/BS 11.25e-6 at 1.7B; Table 2;
   IFEval 67.7 / GSM8K 51.6.
4. 2501.12948v2 — DeepSeek-AI, *DeepSeek-R1*, 2025-01-22. 800k distillation samples; Table 5
   (R1-distill-1.5B: AIME24 28.9, MATH-500 83.9).
5. 2409.12122v1 — Qwen Team, *Qwen2.5-Math*, 2024-09-18. Table 5: Qwen2.5-Math-1.5B-Instruct
   AIME24 3/30, AMC23 24/40.
6. 2503.16219v2 — *RL for Reasoning in Small LLMs: What Works and What Doesn't*, 2025-03-20.
   1.5B, 7k samples, 4×A40 24 h, $42; Tables 1–2; degradation after 200 steps.
7. 2505.24864v1 — *ProRL*, 2025-05-30. 1.5B, 2k+ steps, 16k H100-h; §3.2.
8. 2512.16649v1 — *JustRL*, 2025-12-18. 1.5B, 32×A800 × ~15 days; 54.9/64.3 avg; Table 2.
9. 2605.30478v1 — *Improving Small LMs for Code Generation with RLVR*, 2026-05-28. Qwen3-0.6B
   / Llama3.2-1B, MBPP +14.4 pp, Ruff-only −2.0 pp; Table 1.
10. 2609.28765v1 — *RLVR for Small Search Agents*, 2026-09-23. Qwen3.5-0.8B, 0.092 → 0.352 EM,
    3 seeds; Table 1.
11. 2606.04466v1 — *Learning What to Learn*, 2026-06-03. Qwen2.5-0.5B / Llama3.2-1B,
    SFT-then-RL stage-specific data; Table 1.
12. 2611.02574v1 — *RLVR Sharpens the Pencil, Not the Mind*, 2026-11-02. OLMo-2-1B, GSM8K
    58.0 → 61.3, pass@256 94.5 → 96.9; probes invariant.
13. 2606.22189v1 — *L20-Edu-135M*, 2026-06-20. GSM8K 1.82% → 1.59% / 1.21%; single run.
14. 2512.13961v2 — *Olmo 3*, 2025-12-15. Table 15 stage deltas (7B Think); §4.1.1 variance;
    §3.5.4 special-token leakage; §3.6.4 packing.
15. 2504.13837v5 — Yue et al., *Does RL Really Incentivize Reasoning Capacity...*, 2025-04-18.
16. 2412.09871v1 — *Byte Latent Transformer*, 2024-12-13. Table 1; no SFT/instruct experiments
    (grep of HTML v1: 0 SFT, 1 "Instructions" boilerplate, 1 "fine-tuned").
17. 2510.16987v1 — *Back to Bytes: Revisiting Tokenization Through UTF-8*, 2025-10-19. §3.1,
    Table 1 (C0 control bytes, `apply_chat_template`); no SFT results.
18. 2506.19004v2 — *Broken Tokens?*, 2025-06-23. 93.4% random / 90.8% char-level retention;
    +14% char-level on string manipulation/code.
19. 2411.00640v1 — Miller, *Adding Error Bars to Evals*, 2024-11-01. SE formula.
20. 2402.14992v2 — *tinyBenchmarks*, 2024-02-22. 100 curated MMLU examples.
21. 2506.04178v2 — *OpenThoughts3*, 2025-06-04. 1.2M examples, 7B, AIME25 53 / LCB 51 /
    GPQA 54.
22. 2502.03387v3 — *LIMO*, 2025-02-05. 817 examples, AIME24 63.3.
23. 2305.11206v1 — *LIMA*, 2023-05-18. 1,000 examples.
24. 2310.16944v1 — *Zephyr*, 2023-10-25. UltraFeedback 200k, dSFT + dDPO.
25. 2610.22019v1 — *Priming Base Models for Agentic Post-Training*, 2026-10-15. 30B, 500B
    tokens, SWE-V +3.8 with traces; Table 2 (via the repo's prior read, re-used with id).
26. 2610.34894v1 — *What Outcome RL Buys*, 2026-10-26. `p(1−p)` starvation, App. B.2
    (via the repo's prior read, re-used with id).
27. 2610.22236v1 — *Training-Free On-Policy Distillation*, 2026-10-15. Loop filter, §3.4
    (via the repo's prior read, re-used with id).

Repo measurements (this tree):

- `2026-11-07-architecture-frontier.md:242` — 60 kB/s at 95M; 1 GB ≈ 4.6 GPU-h.
- `2026-11-08-efficiency-frontier.md:156` — 1.519 BPB at 246 MB, step 10000.
- `.bulba/goal.md` — 95,035,200 params, 12 layers, 1M-context design.

---

## 10. Open questions

1. **No published byte-level SFT exists.** BLT did none; UTF8Tokenizer defines a protocol but
   reports no SFT numbers. The first byte-level SFT ablation (control bytes vs literal
   delimiter, patch-aligned vs not) is unclaimed territory.
2. **No ≤1B DPO/ORPO ablation.** Tulu 3's +4.0 avg is 8B; SmolLM2 uses DPO at 1.7B but never
   ablates it. The 1B DPO delta is unknown.
3. **No measured transfer of R1-style long-CoT distillation below 1.5B.** The 1.5B point is
   the floor of the published evidence; 95M is 16× below it.
4. **No published compute for most SFT papers.** Tulu 3, SmolLM2, SmolTulu, R1-distill and
   OpenThoughts3 report data and hyperparameters but not GPU-hours, so pp/GPU-h cannot be
   computed from the papers alone.
5. **The patching boundary is unmeasured.** No paper reports the cost of a prompt/response
   boundary inside a byte patch; the 0.5–2 pp estimate in S3 is an inference.
6. **Benchmark noise at 1B is unpublished.** Olmo 3's variance table is 7B; the 0.6B MBPP
   SD (1.0–1.6 pp) is the only sub-1B number found. A 1B noise study would be a cheap,
   useful contribution.
