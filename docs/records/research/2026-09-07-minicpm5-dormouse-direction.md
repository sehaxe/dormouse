# MiniCPM5-2B и направление dormouse — глубокое исследование

Дата: 2026-09-07. Источники проверены прямыми заходами (HF model card, arxiv, репорты).

## TL;DR

MiniCPM5-2B реален и свеж (OpenBMB, сентябрь 2026): 2.5B dense, Apache-2.0, avg **53.9** по 34 бенчмаркам — SOTA в классе 2B и выше всех 4B в наборе сравнения. Открыл **все чекпоинты** (Base/Midtrain/SFT/final + DSpark draft) и **весь тренировочный корпус** (UltraData: pretrain/SFT/RL). Данные — **только EN+ZH, русского нет** — для нашего RU-чат-бота напрямую годится pretrain-часть и рецепты, но не SFT/RL-часть. Главная практическая находка: **MiniCPM5-2B влезает в наши 16 GB (BF16 ≈ 5 GB) и может работать frozen-учителем** (KD/JEPA/OPD) для dormouse — это готовый путь к качеству без смены архитектуры. Рецепт post-training (tiered data → deep-thinking SFT → RL-учителя → OPD-слияние) — проверенная версия того, что набросано в docs/architecture/post-training.md.

## 1. Что такое MiniCPM5-2B (проверено)

| Параметр | Значение |
|---|---|
| Параметры | **2,516,756,480** dense (1.98B без эмбеддингов) [1] |
| Архитектура | стандартный `LlamaForCausalLM`, 42 слоя, GQA 16Q/2KV [1] |
| Контекст | 131,072 [1] |
| Лицензия | Apache-2.0 (веса и данные) [1] |
| Результаты | avg **53.9**/34 бенча; LiveCodeBench v6 **69.1**, AIME-2025 **86.5**, MMLU-Pro **70.8**, NoLiMa **68.1**, SWE-bench Verified **46.4** — обходит Qwen3.5-4B (51.1), granite-4.2-3B и LFM2.5-8B-A1B [1] |
| Чекпоинты | Base → Midtrain → SFT → final (RL+OPD) + **DSpark draft** [1] |

Устройство качества по их card: stable+decay base → mid-training → **400B токенов deep-thinking SFT** → RL-учителя по доменам (математика/код/агенты/письмо, JustRL II critic-based) → **OPD** (on-policy distillation: full-vocab reverse KL от логитов 16 RL-экспертов, промпты переиспользуются). RL+OPD даёт +10.96 среднего по reasoning, +6.96 по агентам [1].

## 2. Открытые данные (UltraData)

- **Ultra-FineWeb** (1.29B строк) и **Ultra-FineWeb-L3** — L3-рафинированный веб: **400B+ EN + 200B+ ZH токенов**, Q&A-pair generation + multi-style rewriting [2].
- **UltraX** — качественный web-pretrain датасет; на FineWeb-бенче 16B токенов UltraX обгоняет raw/ProX-C при 20B [3].
- **UltraData-Code** (L0–L3 тировая кодовая дата), **UltraData-Math** [1].
- **UltraData-SFT-2605** (12.2M строк; 400B токенов deep-thinking SFT), **UltraData-SFT-Agent-2609** (500K агентных сэмплов), **UltraData-RL-2609** (80K+ RL-сэмплов: математика, код, знания, long-context) [1].

**Критическая оговорка: языки EN+ZH.** HF-теги модели — только English/Chinese; L3-датасет описан как EN/ZH. Русского нет ни в одном датасете.

## 3. Что это значит для dormouse

### 3.1 Данные
- **Pretrain**: Ultra-FineWeb/UltraX добавят качество в общий микс (байтовая LM ест любой текст), но русский контент dormouse тянет из своих источников (rudialog 6.7 GB, ruweb 19 GB — уже на диске). Польза UltraData для RU-бота — косвенная (качество рассуждений в миксе).
- **SFT/RL**: напрямую неприменимы для русского чата. Переносится **метод**: тировая дата-менеджмент (arxiv 2602.09003 [1]) и структура deep-thinking SFT. Русскую SFT-дату придётся собирать (существующие RU-инструктивные корпуса + перевод/переписывание UltraData-SFT как один из источников).

### 3.2 Учитель — главный ход
MiniCPM5-2B BF16 ≈ 5 GB — **влезает в 16 GB рядом со student'ом** (small=7.5M даже base=12M). Это открывает:
- **KD по логитам** на русском тексте: teacher даёт распределение — student учится; это переносит «ум» без огромного корпуса.
- **OPD-схема** как в оригинале: reverse KL по ответам — рецепт опубликован и открыт.
- Апгрейд собственного JEPA: сейчас teacher = EMA-копия student (внутренняя цель); добавить cross-model JEPA/KD от MiniCPM5 — стабилизатор представлений, ровно то, что рекомендует и alphaxiv-анализ ( frozen teacher → logits + hidden targets).
- DSpark: OpenBMB выпустила draft-модель того же класса, что наш aux-хед — направление валидировано индустрией.

### 3.3 Луп и PonderNet — что говорят проверенные работы
- **STARS** (arxiv 2605.26733 [4]): луп-модели деградируют за пределами обученной глубины (Ouro SFT: 70.5% @4 → 53.0% @8); лечение — приближение к устойчивой неподвижной точке через **Jacobian Spectral Radius Regularization + random loop sampling**. Прямо релевантно нашему обязательному max_iter=48: **random-depth training стоит сделать обязательным**, иначе 48 итераций будут хрупкими.
- **RL-Halting / learned stochastic stopping** (arxiv 2606.29983 [5]): наивный PonderNet-объектив нестабилен (OOD acc 29.4±15.1 против 45.0±2.7 у RL-halting с сэмплированной глубиной). Не «выбросить PonderNet», а обучать остановку на сэмплированных глубинах, не только через взвешенную сумму.
- **Ouro/LoopLM** (2510.25741): entropy-регуляризация adaptive exit обязательна, иначе exit схлопывается [6].
- Вывод для M3+: в fused-кернел закладывать не только фиксированный N=48, но и случайную глубину на batch (это дёшево на host-стороне: просто меньше итераций в цикле запусков) + entropy-регуляризацию halt-головы (уже есть KL — проверить веса).

### 3.4 alphaxiv-анализ: сверка с реальностью
Тезисы анализа проверяемы и в основном подтверждаются (STARS/RL-Halting/Engram placement — реальные работы, цифры совпадают). Но по отношению к dormouse он местами описывает то, что УЖЕ есть: наш контроллер и есть «умный роутер KDA/MSA» (sigmoid-blend, не hard-routing), Engram уже встроен в луп (анализ рекомендует ранний вход — это реальное расхождение, требующее A/B), ternary-SCT уже наша технология. Что_analysis недооценивает: простота кода — половина наших требований; ByteFlow-chunker, per-token halting, иерархический роутинг — большие подсистемы против принципа минимальности.

## 4. Рекомендованное направление

1. **Сейчас**: закончить fused-ядра (M3–M6) — без них 48 итераций нежизнеспособны, а всё остальное ортогонально.
2. **Параллельно, дёшево**: поднять MiniCPM5-2B локально как teacher; прототип KD-лосса (логиты на batch, reverse KL) рядом с существующим JEPA/DSpark.
3. **После fused**: претрейн chat1 на 48 итерациях + random-depth; данные — rudialog + подмес Ultra-FineWeb-L3 EN/ZH для рассуждений.
4. **Пост-трейн по рецепту MiniCPM5**: RU deep-thinking SFT (собрать/перевести) → RL-учителя по доменам (docs/architecture/post-training.md уже это скелетит) → OPD-слияние от MiniCPM5-экспертов где применимо.
5. **A/B из анализа, которое стоит взять**: Engram на ранний вход vs в лупе (один прогон), random-depth vs фиксированный N.

## Ссылки
[1] https://huggingface.co/openbmb/MiniCPM5-2B (card, бенчмарки, recipe, список датасетов)
[2] https://huggingface.co/datasets/openbmb/Ultra-FineWeb (L3: 400B+ EN / 200B+ ZH)
[3] https://huggingface.co/datasets/openbmb/UltraX-Preview
[4] https://arxiv.org/abs/2605.26733 (STARS)
[5] https://arxiv.org/abs/2606.29983 (Learned Stochastic Stopping / RL-Halting)
[6] https://www.alphaxiv.org/abs/2510.25741 (Ouro/LoopLM)

## 5. Синтез: архитектура dormouse v2 («интеллект на гигабайт»)

Дополнительные доказательства (раунд 2):
- **BLT** (arxiv 2412.09871, Meta): байтовая LM с динамическим патчированием сравнялась с Llama-3-токенайзером на масштабе; статические патчи хуже, чрезмерное сжатие (BLT-Space) проигрывает — окно 2-3x безопасно [7].
- **Memory Layers at Scale** (arxiv 2412.09764, Meta): memory-слои до 128B параметров памяти на 1T токенов — **обгоняют MoE при равных FLOPs**; это независимое подтверждение оси Engram как главного носителя ёмкости [8].
- STARS + RL-Halting (см. §3.3): фиксированная глубина хрупка; random-depth + регуляризация стабильности — обязательны.

### Итоговый блупринт (против текущего кода)

| Блок | v2 | Сейчас в коде | Действие |
|---|---|---|---|
| Луп | shared block, T random 1..8 + PonderNet-halt на сэмплированных глубинах | фиксированный max_iter 8/12 | смягчить; 48 отменён |
| Engram | один ранний модуль (после embedding), таблицы в RAM | внутри каждого loop-а | A/B размещения |
| Эксперты | **hard top-1 + shared expert**, общие polar-базы + тернарные ядра экспертов | soft-blend 3 TSCT (считаются ВСЕ) | переделка blend → routing |
| JEPA/KD | **оффлайн-таргеты** от MiniCPM5-2B (5 GB BF16, влезает рядом) | онлайн EMA-teacher (второй forward, жрёт VRAM) | самое дешёвое+эффективное |
| Attention | KDA основа + MSA бюджет (10-25%) | уже есть (контроллер-blend) | добавить бюджет-режим позже |
| Компрессия байтов | ByteFlow-chunker — **отложить** до контекстов 4K+ | нет | не сейчас (BLT: выигрыш на длине) |
| Веса | ternary-SCT FFN (уже есть), роутер/QK/retention — точные | уже так | держать |

Фазы: P0 fused M3-M6 на N_max=8 → P1 chat1 (small) с оффлайн-JEPA (VRAM снова влезает batch 10) → P2 A/B: Engram-размещение, top-1 MoE → P3 1B-total MoE-SCT (active 150-250M) → P4 пост-трейн по рецепту MiniCPM5 (RU SFT → RL-учителя → OPD).

[7] https://arxiv.org/abs/2412.09871  [8] https://arxiv.org/abs/2412.09764

## 6. DeepSeek-V4.1-Flash tech report + Looped Flows (раунд 3, 2026-09-13)

Источник: DeepSeek_V41_Tech_Report.pdf (HF), §2.4-2.5; arxiv 2609.11801.

**Архитектура V4.1-Flash**: 40 слоёв = 20 causal-encoder + 20 decoder (CED); SWA+global в каждом слое; 552B backbone + **196B Engram**; 8B активных на prefill / 16B на decode; Single-Pass mHC; DSpark вместо MTP; FP4 main-KV (QAT).

### Прямо переносимое в dormouse (по убыванию ценности)

1. **Sinkhorn-momentum вместо Adam для Engram-таблиц** (§2.5): momentum-буфер (×1 память) вместо Adam m+v (×3), «эмпирически обгоняет Adam»; применяется также к token embedding и prediction head. Наш host-Adam (offload.rs) ест ×3 RAM — замена удваивает ёмкость таблиц в той же памяти. Nesterov, без weight decay.
2. **Head-wise Muon для Q/K** (§2.5): веса разбиваются по головам ПЕРЕД ортогонализацией — подтверждено на GLM 5 и Kimi-K3. Это закрывает открытый пункт AGENTS.md («split fused params before orthogonalization — fusing mixes singular directions»). Дешёвый апгрейд optim.rs.
3. **Engram: слои 1 и 14, не в каждом слое/лупе** (§2.4.2) — независимое подтверждение рекомендации alphaxiv против нашего in-loop размещения. Их дизайн: multi-head hashing (8 голов), orders {2,3,4}, 2048 dim/order, ~16M записей на голову, размеры таблиц — **различные простые числа** (гигиена коллизий), таблицы FP8, короткий causal-conv опущен («gain не оправдывает сложность» — независимое подтверждение нашего revert use_short_conv!).
4. **DSpark: отдельная стадия после претрейна** (§2.4.3), backbone frozen, в пост-трейне без градиентов в backbone. Мы тренируем его совместно — стоит перенести на стадию.
5. **Prefetch с host-memory на инференсе** (RDMA, перекрытие с compute) — валидация нашего host-RAM offload.
6. **FP4 main-KV (E2M1 + scale/16 каналов, QAT после RoPE, SWA-KV остаётся FP8)** — для будущего инференса dormouse.

### Looped Flows (2609.11801) — не DeepSeek, но по теме лупа

Обучение рекуррентности **локальными denoising-целями** (убывающий шум + общий шум) вместо BPTT через все лупы; градиент идёт через несколько апдейтов, но состояния учатся переносить вычисления. SOTA looped-моделей: **58.8% ARC-AGI-1, 12.2% ARC-AGI-2**. Для dormouse — кандидат на целевую функцию лупа вместо полного unroll'а: дешевле по памяти графа и решает ту же проблему «ранние лупы должны готовить поздние», что STARS. Исследовательская опция для M3+, не сейчас.

### Сверка с нашим bluprint'ом v2
- П.1-3 меняют план: Sinkhorn-оптимизатор таблиц — в offload.rs; head-wise Muon Q/K — в optim.rs; Engram-placement A/B — приоритет подтверждён дважды.
- CED/CSA2/иерархический indexer/FP4 — serving-штуки больших моделей, в dormouse не переносим (принцип минимальности).
