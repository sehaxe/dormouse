<p align="center">
  <img src="assets/banner.svg" alt="dormouse — байтовая Mixture-of-Experts языковая модель, обученная на одной потребительской видеокарте" width="1200">
</p>

# 🐹 dormouse

[![ci](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml)

[English](README.md) · **Русский**

Байтовая (byte-level) Mixture-of-Experts языковая модель, обученная от начала до конца на **одной потребительской видеокарте** (RTX 5060 Ti, 16 ГБ, $429). Без токенизатора, без кластера, без учителя-дистиллятора — весь рецепт исполняется на карте, и каждое утверждение в этом репозитории несёт своё измерение.

Rust/burn-оригинал, из которого всё выросло, живёт на ветке [`rust-legacy`](../../tree/rust-legacy). `main` — Python-проект.

**Содержание** — [Что это](#что-это) · [Результаты](#результаты) · [Быстрый старт](#быстрый-старт) · [Архитектура](#архитектура) · [Документация](#документация) · [Лицензия](#лицензия)

## Что это

- **Байтовый уровень**: словарь из 256 байтов. Никакого BPE и токенизаторного налога на русский, код или что угодно, чего токенизатор не видел на обучении.
- **Mixture-of-Experts с dropless-роутингом**: 1024 эксперта, top-k 2, без capacity-дропов — каждый токен сохраняет всех экспертов, к которым его направил роутер.
- **KDA linear attention** (chunked gated delta rule в стиле Kimi, через FLA): контекст O(n) и фиксированное рекуррентное состояние 256 КБ на **любой** длине — KV-кэшу расти некуда.
- **TSCT low-rank эксперты**: каждый эксперт — это `U·diag(s)·Vᵀ` (rank 64); именно это позволяет модели на 1.07B параметров помещаться, учиться и выпускаться на 16 ГБ.
- **Engram-память**: 402M «параметров» хэш-таблиц, которые живут в оперативной памяти, вне вычислительного графа.
- **Weight-shared loop**: три прохода через один и тот же блок на токен (замеренный оптимум; четыре — ничья, две — хуже).
- **Весь пайплайн в одном CLI**: `train → bench → sft → rl → serve`, плюс база знаний, отрендеренная из собственных документов репозитория.

Пресеты: `small` (9.5M, полигон для A/B), `medium` (138M, текущая рабочая лошадка), `b1` (1.07B total / 57M active, флагман).

## Результаты

Измерено, датировано, воспроизводимо — числа ниже отражают состояние на 2026-10-19.

| что | число |
|---|---|
| **претрейн medium** (`mix1`, микс веб+вики+рус+матем) | **BPB 1.936** на held-out (окно 196 608 Б) после 1.47 ГБ увиденного |
| скорость, medium (b24 × s1024) | **40.9k токенов/с**, ~12 ГиБ пик, одна RTX 5060 Ti |
| скорость, b1 (b8 × s512) | ~8k токенов/с eager, 13 ГиБ |
| первые downstream-числа (zero-shot, n=1000, случайное угадывание = 25%) | HellaSwag **30.5%**, ARC-Easy **27.2%**, synth-math 24.0% — при 0.37B увиденных токенов |
| тесты | **861 тест**, ruff чистый |

Строку HellaSwag стоит перечитать: она впереди GPT-2-small (31.4%) при **в 24 раза меньшем** числе обучающих токенов и впереди Pythia-160M (30.3%) при ~800-кратно меньшем.

Каждый механизм в дереве был допущен 3-сидовым A/B на фиксированном протоколе, и проигравшие записаны, а не тихо удалены: фикс ColRow (−0.043 BPB, крупнейшая единичная победа), глубина 3 (−0.007), dropless-роутинг, Triton Newton–Schulz (52 диспатча → 1), CUDA-graph оптимизатора (−11.7 мс/шаг), bf16-моментум Muon (−1.23 ГиБ), Engram в оперативной памяти (бит-в-бит идентично). Отклонены по замерам: NorMuon (+0.072), schedule-free Muon (+0.043), SIGReg (+0.148), LOOM (ничья), MSA на коротком контексте (ничья), тернарные веса с нуля (+0.04), KDA recompute-флаг (ничья).

## Быстрый старт

Нужен [uv](https://docs.astral.sh/uv/) и Python 3.12.

```bash
uv sync                # CPU-окружение; тесты бегут здесь
uv sync --extra gpu    # замеренный стек: torch 2.13.0+cu129 + FLA (sm_120)

uv run pytest                       # сюита (861 тест; CUDA-гейты сами скипаются)
uv run ruff check .                 # линт

# обучение модели на байтовом корпусе
uv run dormouse train medium --data corpus.bin --eval held.bin \
  --device cuda --steps 60000 --batch 24 --set seq_len=1024 \
  --set engram_ram=true --set moe_capacity=0 --set moe_kernel=triton \
  --ckpt-name mix1 --out-dir checkpoints/mix1

uv run dormouse bench --ckpt-name mix1 --out-dir checkpoints/mix1 \
  --tasks arc_easy,hellaswag --limit 1000     # downstream, байтовый скоринг

uv run dormouse sft medium --data sft.bin --build-sft \
  --chat-data pairs.jsonl --init-from checkpoints/mix1/mix1.pt \
  --ckpt-name sft11 --out-dir checkpoints/sft

uv run dormouse serve --out-dir checkpoints/sft --ckpt-name sft11 --port 3783
```

`dormouse --help` перечисляет все команды; `dormouse info <preset>` печатает таблицу бюджета параметров. Каждая ручка — типизированный флаг или `--set key=value`, который отказывается загружаться на неизвестном ключе: конфиг, который врёт, — это баг, и схема строга намеренно.

## Архитектура

```
src/dormouse/
├── config.py        строгая схема; неизвестные ключи и не-портированные армы отказывают громко
├── instrument.py    слой деградаций LOUD / COUNTED / SILENT
├── model/
│   ├── model.py     DormouseModel: embed → loop → norm → head
│   ├── loopblock.py weight-shared loop; адаптивная глубина; MSA-гейт
│   ├── kda.py       KDA-внимание (chunked ядро FLA; SDPA-фоллбек, считается)
│   ├── experts.py   dropless MoE: роутер, LFB-балансировка, padded/Triton диспатч
│   ├── tsct.py      факторы экспертов U·diag(s)·Vᵀ + storage-лестница (fp8→тернарный)
│   ├── engram.py    хэш-память, в VRAM или в оперативке (бит-в-бит идентично)
│   ├── byteflow.py  байтовый патчер (local→global→decoder)
│   └── sparse/      MSA: block-sparse внимание + fused Triton gather
├── train/           цикл, WSD-расписание, Muon+AdamW, clip, SFT, потоковый микс
├── post/            верификаторы (код/математика), роллауты, CRR-advantage, RL-степпер
├── bench/           downstream-eval: ARC-Easy, HellaSwag, synth math (байтовый скоринг)
├── selfplay/        zero-data self-play ядро (UTM + reward по выравниванию градиента)
└── cli.py           train · bench · sft · rl · serve · decode · plot · info
```

Три вещи написаны свои, потому что ничего такого нет в экосистеме: квантованные TSCT-эксперты, host-таблицы Engram и патчер ByteFlow. Всё остальное — внимание, оптимизатор, форматы квантования, fused-ядра — лучшие доступные зависимости; места, где выбор был сделан против собственной реализации, записаны в `docs/`.

## Документация

База знаний собирается из собственных документов репозитория и деплоится вместе с сайтом:

- **Сайт документации** (`docs-site/`): Starlight-сайт поверх `docs/`, `research/` и рабочих заметок; `npm run build` в `docs-site/` рендерит его локально.
- **`docs/decisions/`**: датированные планы и ADR — карта проекта лежит в 1B-readiness-плане.
- **`docs/research/`**: литературная работа; каждое утверждение несёт свой arXiv id.
- **`docs/reviews/`**: аудиты, включая те, что нашли наши собственные баги.
- **`AGENTS.md`**: свод правил, которым этот репозиторий действительно управляется.

## Лицензия

MIT — см. [LICENSE](LICENSE).
