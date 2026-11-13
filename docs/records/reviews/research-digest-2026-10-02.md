# Три бумаги от research-агента владельца — разбор против нашей архитектуры

Обновлено 2026-10-02. Источники: `docs/papers/{context-lms-2609.37725,looped-dit-2609.40305,loopcd-2610.02185}.pdf`
(provenance.tsv). Все три — про вес-общие/цикловые модели: наша LoopBlock-семья.

## 1. Looped Diffusion Transformer (2609.40305) — ПРЯМЫЕ ДЕЙСТВИЯ

**Контекст**: weight-shared middle blocks внутри denoising-шага; «наивный лупинг
не работает стабильно» — нужны два ремонта.

| ремонт бумаги | у нас | действие |
|---|---|---|
| **Deep Supervision** — каждый промежуточный loop декодируется через shared head и учится на общей цели | **УЖЕ ЕСТЬ**: L_Rec — per-iteration CE внутри цикла (`model.rs`, §3.8) | валидировано бумагой — не трогать |
| **Self-Modulating Attention (XSA)** — повторный attention «эродирует» локальную информацию; фикс: удалить компонент attention-обновления вдоль собственного value-направления токена (parameter-free, только внутри looped-блоков) | **НЕТ** — наш KDA-гейт работает как обычный attention все 4 итерации | **F13: реализовать XSA-вариант для KDA-гейта** (parameter-free, за флагом, гейт: динамика повторных итераций) |

Числа бумаги: 260M looped > 1.7B non-looped (71.5 vs 69.0), 4.9× меньше inference
FLOPs; matched-compute: looped 59.1 vs deeper non-looped 58.1. Область: diffusion
T2I, 260M масштаб — перенос на LM/KDA требует нашего A/B (F13-гейт).

## 2. LoopCD (2610.02185, Apple) — inference-рычаг БЕЗ тренировки

**Контекст**: каждый промежуточный loop — уже декодируемое предсказание; ранний
loop = «amateur» для contrastive decoding. Two варианта: LoopCD-Logits (один
лишний output-pass) и **LoopCD-Hidden (нулевые накладные)** — комбинирует
скрытые состояния до output-слоёв.

| результат бумаги | к нам |
|---|---|
| Ouro-2.6B AIME pass@1: 61.88 → **73.33** (adaptive LoopCD-Logits) | наш dec/generate путь: depth-4 loop → LoopCD-Hidden по итерации 2 |
| **Huginn на 16 итерациях ≥ 32-итерационный baseline** → −46% FLOPs | **наша depth-4 модель может инференсироваться на depth-2 без потери** — тренировка не нужна |
| GSM8K mixed; Looped-Qwen3 pass@1 падает | рисковая зона: мерить на нашем генераторе до включения |

**F14: LoopCD-Hidden в generate/serve** — decode-only, тренировка не требуется,
риск ограничен инференсом. ADR-0020-оговорка: FLOP-экономии — арифметические.

## 3. Context Language Models (2609.37725) — ответ на E1 (пост-тренинг)

Модель управляет своим контекстом как файлом ( unrestricted edits), обучение
управлению через RL: BrowseComp-Plus +11.4% точности при −21.5% FLOPs; 47.6%
прирост от online RL (Qwen3.5-9B). **Для нашей RLVR-главы** (post-training.md):
context-management как RLVR-способность — архитектурных требований к претрейну
не предъявляет (обычная LM), но пайплайн «skill-optimization loop» стоит
записать в план пост-тренинга.

## Вердикты

- **F13 (XSA в KDA-цикл)**: реализация parameter-free, гейт: динамика повторных
  итераций + A/B волна. Приоритет — после посадки fidelity-fix.
- **F14 (LoopCD-Hidden в generate)**: decode-only, бесплатный по тренировке.
  Приоритет — любой момент; измерить на нашем генераторе.
- **E1**: CLM-паттерн → `docs/architecture/post-training.md` дополнение (владелец).
- **A1 частично отвечен**: семейство weight-shared валидировано индустрией
  (Hyperloop, SMELT scaling laws, Training-Free Looped Transformers — все 2026);
  два известных ремонта лупинга — deep supervision (есть) + регуляция attention
  (нет, F13).

## Ограничения честности

Все три результата — в своих областях (diffusion T2I; four looped families;
agent-задачи). Прямой перенос на byte-LM/KDA-цикл — гипотеза до нашего A/B.
