# Super-research: каждая технология модели, ранжированные свопы под FLOP-эффективность

**2026-10-07, research pass.** 8 раундов discovery через arXiv MCP, ~60 кандидатов,
25+ верифицированных абстрактов, 3 бумаги прочитаны по секциям (FactorEngram, GSA2, RMT).
Цель: максимальный интеллект на FLOP на 16 GB VRAM + launch-bound бэкенд.

**Ключевой фрейминг.** Наш GPU launch-bound: 13.3% средняя утилизация, 79% сэмплов ≤5% [AGENTS §3.1].
FLOP-сокращение ≠ wall-clock сокращение, если число ядер не падает. RMT — идеальная иллюстрация:
−58% FLOP к тому же лоссу, но +43% wall-clock на фиксированном бюджете токенов [1]. Поэтому ранжирование
разделяет (a) интеллект-на-FLOP (качество при фиксированном вычислении) и (b) FLOP-на-токен
(адаптивная глубина, кванты, sparse attention).

**Блокирующее условие.** Бисект численного дефекта 0.22 (KDA-арм = источник спайков, no-kda 3/3 чисто)
должен закрыться до любого A/B [AGENTS §3.4]. Всё ниже — после здоровой базы.

---

## TL;DR — ранжированные свопы

| # | Своп | Технология | Ожидаемый эффект | Миграция | A/B гейт |
|---|------|-----------|-----------------|----------|----------|
| 1 | **FactorEngram** → Engram | basis-факторизация таблицы памяти | +43.3 pp NIAH-3 vs Engram при 340M [2] | средняя | engram_rows ladder |
| 2 | **GSA2** → KDA | two-sided associative memory | +2.59 common-sense avg при 1.3B [3] | средняя | после бисекта |
| 3 | **ScheduleFree+** → WSD | LR-free + schedule-free | +31% при 1000 tok/param [4] | низкая | 3 сида, 2k шагов |
| 4 | **MoR/rand-depth** A/B | token-level adaptive depth | FLOP-экономия на простых токенах | уже в коде | depth 2 vs 4 |
| 5 | **8-bit Muon states** | квантование состояния оптимизатора | −62% optimizer memory [5] | низкая | parity гейт |
| 6 | **SUMO** → NS-3 ретракция | exact SVD vs Newton-Schulz | меньше ошибка аппроксимации [6] | средняя | max_ortho latch |
| 7 | **SALT/frozen teacher** → EMA | замороженный учитель для JEPA | −1 forward per step [7] | низкая | aux parity |
| 8 | **FP8 recipe** | end-to-end FP8 | −22% time, +19% throughput [8] | высокая | kernel support |

**Парк (не своп):** RMT (−58% FLOP, но +43% time — launch-bound GPU делает хуже) [1];
sparse attention (seq 512 байт — нерелевантно); FP4/Quartet (нет kernel support в нашем стеке);
distillation (post-training lane); EBM heads (не реализовано, playbook §3.6).

**Keep (валидировано):** byte-level (Toppling the Hierarchy [9]); WSD (river valley [10]);
JEPA+KoLeo (LLM-JEPA [11]); DSpark (GrowMTP — post-training [12]).

---

## 1. Память: Engram → FactorEngram

**Текущее:** FNV-хеш, ORDERS=[2,3,4], 25k rows, scalar gate с lam_max floor.

**FactorEngram** [2] (Baidu, 2609.35578): факторизованные коэффициенты над словарём базисных
векторов, общих для всех паттернов; basis-level contextual gating; sparsity regularization (λ=1e-3).
Критика нашего дизайна точечно: «Engram treats each retrieved embedding as a monolithic unit...
polysemous patterns cannot selectively read out the components... parameters are shared only
through hash collisions, which are largely unrelated to semantics».

**Результаты (340M, 30B tokens FineWeb-Edu, 8K ctx):**
- WikiText PPL: 23.19 → 21.29 (Transformer → FactorEngram)
- LAMBADA PPL: 24.90 → 21.69
- NIAH-3: 15.0 → 58.3 (+43.3 pp vs Engram)
- NIAH-2: +41.5 pp vs Engram
- 1B (120B tokens): retrieval +9.3..27.6 pp over Engram

**X-GRAM** [13] (2604.21724): data-aware n-gram extraction, frequency-aware compression,
SwiGLU ShortConv refinement, depth-aware gating; +4.4 avg acc vs vanilla hashing при 0.73B/1.15B.

**Вердикт:** FactorEngram — прямой апгрейд нашей таблицы. Миграция: формат таблицы + gate rewrite
в vendor crate. A/B: существующая engram_rows ladder. Ожидание: лучшее качество на те же params
(basis-разделение = эффективная ёмкость на slot выше).

---

## 2. Attention: KDA → GSA2

**Текущее:** KDA (FLA softplus decay, kdafix2), fused OFF по умолчанию, backward не работает на 0.22.

**GSA2** [3] (2610.02816): two-sided associative memory correction — Gated Oja Rule (key-side)
+ Gated Delta Rule (value-side), shared slots, hardware-efficient chunkwise training.

**Результаты (1.3B, 100B FineWeb-Edu):**
- Common-sense avg: GSA2 31.26 (recurrent) / 36.70 (hybrid) vs KDA 28.67 / 35.73
- GDN2: 29.88 / 35.41
- LM стабилен; recall улучшается

**Hybrid analysis** [14] (2507.06457): 72 модели, 340M/1.3B, 6 linear variants × 5 hybrid ratios.
LM стабилен на всех соотношениях; recall улучшается с ростом full-attention слоёв,
особенно ниже соотношения 3:1. Selective gating + hierarchical recurrence важны.

**Вердикт:** GSA2 — candidate swap для attention arm. Но KDA сломан на 0.22 (бисект) —
своп только после здоровой базы. Hybrid ratio: наш per-iteration controller blend уже
покрывает placement; 3:1 ratio — ориентир для A/B.

---

## 3. Loop / Depth: MoR, Adaptive Depth, Fully Looped

**Текущее:** LoopBlock weight-shared, fixed depth (ADR-0013), --rand-depth и MoR есть, не A/B'ены.

**MoR** [15] (2507.10524): token-level recursive depth, shared stack, quadratic attention
только среди активных токенов. Параметр-эффективность + adaptive compute в одном.

**Adaptive Depth** [16] (2607.20519): fixed-prior depth supervision + post-hoc confidence
readouts match/beat learned gates. Провал learned halting — в trajectory, индуцированной
joint gate training, а не в expressivity gate. Ouro-1.4B/2.6B.
→ **Валидирует наш ADR-0013** (fixed depth won, halt head deleted). Post-hoc confidence readout
— дешёвый инструмент для --eval-depths кривых.

**Fully Looped** [17] (2605.18797): два parameter-free фикса для нестабильности loop
(gradient oscillation + residual explosion): distribute inter-loop signals across all layers
+ attention injection. Стабилизирует до 12 итераций.

**Dynamical phase** [18] (2608.26556): одна архитектура + один objective, разные init →
разные dynamical phases → test-time compute scaling отличается (fold phase τ~N^-2 tail;
Neimark-Sacker убивает его).

**Sparse Token Routing** [19] (2608.20632): **NULL RESULT** — routing даёт negligible accuracy
change vs parameter-matched baselines. Важная честная точка для MoE/token-routing утверждений.

**MELT** [20] (2605.07721): единый KV cache, разделяемый через loop iterations, с learnable gating.

**Вердикт:** MoR/rand-depth — самый дешёвый большой рычаг (уже в коде, нужен только A/B).
Post-hoc confidence readout — бесплатное дополнение к --eval-depths. Fully Looped —
если пойдём в depth 8-16 для overtrained режима.

---

## 4. Residual Stream: RMT (парк)

**RMT** [1] (2506.22696): outer-product residual matrix, residual stream 2.5-4× шире.

**Результаты (46M-405M):**
- К loss 3.03: −58% FLOPs, −25% params, −41% tokens, но +4% time
- На 6B tokens: −28% FLOPs, но +43% time (kernels less hardware-friendly)

**Вердикт:** FLOP-efficient, но wall-clock regressed. Наш GPU launch-bound — меньше FLOP
при том же числе ядер = медленнее. Парк как research, не swap. Если придётся масштабировать
residual stream — hyper-connections или GR дешевле.

---

## 5. Оптимизатор: Muon+ → MiMuon / AdaMuon / 8-bit / SUMO

**Текущее:** Muon+ (Jordan triple, NS-3, ColRow), AdamW fallback, host Nesterov+Sinkhorn.

**MiMuon** [21] (2605.19619): hybrid Muon + momentum SGD. Generalization error O(1/N)
vs O(1/(Nκ^T)) у Muon. Осторожное использование orthogonalization.

**AdaMuon** [22] (2507.11005): element-wise second momentum на orthogonalized updates
+ sign-stabilized orthogonal update. +40% training efficiency claim. RMS-aligned rescaling
— можно переиспользовать LR schedules.

**8-bit Muon** [5] (2509.23106): blockwise quant, parity, −62% optimizer state,
linear quant без dynamic scaling.

**SUMO** [6] (2505.24749): exact SVD moment orthogonalization в adapted subspace.
Верхняя граница ошибки Newton-Schulz через condition numbers.
→ Прямо релевантно нашей NS-3 ретракции (измерено 14.5% от polar при 10:1 spread).

**ScheduleFree+** [4] (2605.19095): LR-free + schedule-free, бьёт WSD,
+31% при 1000 tokens/param, лучше для long training. Model averaging + checkpoint merging.

**WSD dynamics** [10, 23]: river valley landscape, universal across architectures.
У нас уже есть WSD — валидировано.

**Вердикт:** 8-bit Muon — самый дешёвый выигрыш (−62% optimizer memory → больше batch/model
в VRAM). SUMO — если NS-3 ошибка реальна (max_ortho latch). ScheduleFree+ — если идём
в 100k+ шагов. MiMuon/AdaMuon — если generalization bottleneck.

---

## 6. Точность: FP8 / FP4 / f16

**Текущее:** f16 tensor-core alive (91-97% cuBLAS, measured), bf16 невозможен,
act-quant fp4 есть, fp8 factors на sm_120.

**Quartet** [24] (2505.14669): native FP4 end-to-end на Blackwell, low-precision scaling law,
"optimal" recipe. Llama-type models.

**NVFP4** [25] (2509.25149): NVIDIA production pretraining recipe с FP4.

**FP8 recipe** [8] (2509.22536): end-to-end FP8, −22% training time, −14% peak memory,
+19% throughput, lossless vs BF16. Hybrid-granularity quantization.

**Вердикт:** FP8 — если появится kernel support в нашем стеке (cubecl/cubek).
FP4 — нужны ядра, которых нет. f16 — уже работает, самый большой ближайший рычаг
(4.5-4.8× на GEMMs, но GEMMs 2-5% шага — выигрыш ~0.1-0.35% на текущем масштабе,
растёт с моделью).

---

## 7. Aux Heads: JEPA / KoLeo / DSpark

**LLM-JEPA** [11] (2509.14252): JEPA objectives бьют standard LLM training objectives
на Llama3/OpenELM/Gemma2/Olmo, robust to overfitting. → Валидирует наш aux-on default.

**ER-JEPA** [26] (2609.36952): experience replay улучшает JEPA.

**SALT / frozen teacher** [7] (2509.24317): замороженный учитель достаточно (без EMA),
compute-optimal при matched FLOPs. → Наш EMA teacher можно заморозить; --jepa-precompute
уже делает offline latents.

**GrowMTP** [12] (2609.16648): RL выращивает свой draft head, 2.13× rollout speedup.
→ Post-training lane, не pretraining.

**Вердикт:** Заморозить EMA teacher (SALT) — −1 forward per step. ER-JEPA — если replay
дёшево. GrowMTP — когда придём к RL.

---

## 8. Offload: host RAM

**Deep Optimizer States** [27] (2410.21316): interleaved offloading, динамическое
перемещение optimizer state между host и GPU во время fwd/bwd/update phase fluctuations.

**MLP-Offload** [28] (2509.02480): multi-level multi-path offloading, I/O overheads
during update dominate, cache-efficient + concurrency-controlled.

**COAT** [29] (2410.19313): сжатие optimizer states + activation для FP8 training.

**GaLore** [30] (2403.03507): low-rank projection для optimizer states.

**Вердикт:** Наш host-RAM n-gram offload (Nesterov+Sinkhorn) — другой класс (таблицы,
не optimizer states). Deep Optimizer States — если будем offloadить оптимизатор.
COAT — если придём к FP8.

---

## 9. Byte-level LM

**Toppling the Hierarchy** [9] (2609.00463): pure byte-level бьёт hierarchical downsampling
для character manipulation; byte-level attention — механизм.

**Byte-level Simulation** [31] (2604.27263): decouples subword benefits.

**Вердикт:** Валидирует наш byte-level выбор (vocab 256). Keep.

---

## 10. Distillation / Data

**KD vs pretraining** [32] (2404.19319): KD бьёт from-scratch при фиксированном compute,
особенно с repeated data.

**Active data curation** [33] (2411.18674): curation effectively distills.

**Online reweighting** [34] (2605.05227): лучше generalization чем offline curation.

**Internal data repetition destroys LMs** [35] (2606.24998).

**Вердикт:** Post-training lane (docs/architecture/post-training.md). Для pretraining:
online reweighting > offline, если есть бюджет на curation.

---

## 11. Overtraining

**Reusing overtrained LMs saturates scaling** [36] (2510.06548).

**Overtrained LMs harder to fine-tune** [37] (2503.19206).

**Beyond Chinchilla** [38] (2401.00448): inference-aware scaling laws.

**Вердикт:** Для нашей цели (1-10B overtrained на одной карте) — валидирует стратегию.
Но следить за saturation при повторном использовании.

---

## 12. Sparse Attention (MSA re-entry)

**PIVOT** [39] (2607.24593): query-group indexing, training-free, drop-in для DSA indexer.

**IndexCache** [40] (2603.12201), **SpargeAttention** [41] (2502.18137),
**MTraining** [42] (2510.18830).

**Вердикт:** Seq 512 байт — sparse attention нерелевантно. Парк до long-context амбиций.

---

## 13. EBM Heads

**EBT** [43] (2606.23382): energy как predictor reading difficulty — psycholinguistics
application, не основная EBT paper (2507.02092 в AGENTS §3.6).

**Вердикт:** Не реализовано, playbook §3.6. Очередь: energy head на loop state.

---

## 14. Dynamic Compute

**DYNAMAX** [44] (2504.20922): early exits для Transformers/Mamba.

**DTRNet** [45] (2509.00925): dynamic token routing для quadratic costs.

**Sparse Token Routing** [19] (2608.20632): null result (см. §3).

**Вердикт:** MoR уже покрывает token-level dynamic depth. DTRNet — если придём
к full attention на длинном контексте.

---

## Ссылки

[1] 2506.22696 — Residual Matrix Transformers
[2] 2609.35578 — FactorEngram
[3] 2610.02816 — Gated Slot Attention-2
[4] 2605.19095 — ScheduleFree+
[5] 2509.23106 — Effective Quantization of Muon States
[6] 2505.24749 — SUMO
[7] 2509.24317 — SALT / frozen teacher JEPA
[8] 2509.22536 — FP8 Training Recipe
[9] 2609.00463 — Toppling the Hierarchy
[10] 2410.05192 — WSD River Valley
[11] 2509.14252 — LLM-JEPA
[12] 2609.16648 — GrowMTP
[13] 2604.21724 — X-GRAM
[14] 2507.06457 — Hybrid Linear Attention Analysis
[15] 2507.10524 — MoR
[16] 2607.20519 — Adaptive Depth in Looped Transformers
[17] 2605.18797 — Fully Looped Transformer
[18] 2608.26556 — Dynamical Phase Selection
[19] 2608.20632 — Sparse Token Routing (null result)
[20] 2605.07721 — MELT
[21] 2605.19619 — MiMuon
[22] 2507.11005 — AdaMuon
[23] 2601.09000 — Universal Dynamics of WSD
[24] 2505.14669 — Quartet FP4
[25] 2509.25149 — NVFP4
[26] 2609.36952 — ER-JEPA
[27] 2410.21316 — Deep Optimizer States
[28] 2509.02480 — MLP-Offload
[29] 2410.19313 — COAT
[30] 2403.03507 — GaLore
[31] 2604.27263 — Byte-level Simulation
[32] 2404.19319 — KD vs Pretraining Fixed Budget
[33] 2411.18674 — Active Data Curation
[34] 2605.05227 — Online Reweighting
[35] 2606.24998 — Internal Data Repetition
[36] 2510.06548 — Reusing Overtrained LMs
[37] 2503.19206 — Overtrained LMs Fine-tune
[38] 2401.00448 — Beyond Chinchilla
[39] 2607.24593 — PIVOT
[40] 2603.12201 — IndexCache
[41] 2502.18137 — SpargeAttention
[42] 2510.18830 — MTraining
[43] 2606.23382 — EBT Reading Difficulty
[44] 2504.20922 — DYNAMAX
[45] 2509.00925 — DTRNet
