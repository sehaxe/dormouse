# 1B on one 16 GiB card: the memory ledger, the published record, and what breaks

**Date:** 2026-11-08 · **Card:** RTX 5070 Ti, 16 GiB, sm_120, 45.7 TFLOP/s bf16 peak
(repo: `scripts/vram_ledger.py:48`) · **Mode:** CPU-only research, no GPU job (the training
run holds the card). The 1B geometry below is instantiated on the `meta` device.

**Tags.** **[T]** — the paper's own words/table. **[C]** — arithmetic done here, formula
printed. **[S]** — SPECULATION, no direct evidence. **[M]** — measured in this repo, with the
path. Every claim carries one of these; there is no untagged number in this file.

**The model priced (unless stated otherwise).** A vanilla pre-LN decoder of the same family as
`port/configs/flagship3_dense.toml`, scaled to 1B: **L=28, d_model=1792, n_heads=28,
head_dim=64, d_ffn=6656** (the 95M's f/d = 3.69 kept), vocab 256, GELU MLP, SDPA, fp32
residual. **N = 1,028,724,480**, of which Muon owns 1,027,604,480 and AdamW 1,120,000
(`is_muon_param` rule, `port/src/dormouse/train/optim.py:1667-1689`; counts from
`DormouseModel(cfg, device="meta")`). This is the shape that matches our own aspect ratio
(1792/28 = 64, vs 832/12 = 69 and Ripple-65B's 5120/80 = 64).

---

## TL;DR — вердикт по-русски

1. **Память: 1B влезает, но только b=1.** Статическое состояние (веса+градиенты+Muon+AdamW+
   NS+контекст) = **13.25 GiB** при fp32 Muon и **11.34 GiB** при bf16 Muon. Это уже 75-88 %
   из 15 GiB *до* активаций. На b=1/s=2048 с checkpointing и bf16 Muon выходит **11.84 GiB** —
   влезает с запасом 3.2 GiB. На b=1/s=2048 **без** checkpointing — 14.59 GiB (влезает, но
   запас 0.4 GiB, то есть «не влезает» на практике). **b=8 не влезает ни при какой длине
   сегмента:** минимум без нового кода — 15.34 GiB (b8/s2048, ckpt, bf16 Muon), это
   выше рабочего потолка 15.0 и без запаса на фрагментацию. b8/s8192 — 27.4 GiB.
2. **b=8 — это не требование, а привычка.** b=1 + `grad_accum=8` даёт тот же поток байт и тот
   же optimizer step, память — как у b=1. Единственная потеря — чуть меньшая загрузка тензорных
   ядер на микро-шаге; при d=1792 и s=2048 GEMM всё равно большой. Так что «1B не влезает»
   надо читать как «1B не влезает с микро-батчем больше 1».
3. **Сегментирование не спасает vanilla-transformer.** `segment_len` для `block_form=
   "transformer"` **отказан кодом**: «the vanilla decoder has no state to carry (its K/V context
   is the window)» (`port/src/dormouse/config.py:5284-5290`). Нарезка 8K/32K для полного
   внимания — это *другая модель* (окно внимания), а не трюк с памятью. Точный 32K на 1B
   требует KV-кэш **6.13 GiB** (MHA, bf16) сверх активаций; 1M — 187 GiB. Единственный точный
   носитель состояния — KDA/CSA (`docs/reviews/2026-11-08-segment-csa.md`), и он существует
   только в loop-block.
4. **В литературе 1B с нуля на одной карте 16-24 GiB нет.** Ближайшие точки — Pythia-1.0B на
   **64×A100-40G** (4 830 A100-часов) и TinyLlama-1.1B на **16×A100-40G** (≈34 560 A100-часов).
   На нашей карте это 9 и 41-99 лет соответственно. Единственная честная формулировка: такого
   прогона не опубликовано, и причина — не память, а пропускная способность.
5. **Chinchilla бьёт по плану 22 GB.** При 20 токен/параметр 22 GB (=5.5B BPE-эквивалентов)
   соответствуют **275M**, а не 1B. 1B на 22 GB — это 5.5 токен/параметр, в 3.6 раза ниже
   Chinchilla. 80 GB — ровно 20 токен/параметр, то есть 1B на 80 GB — это *в точности*
   Chinchilla-точка. Оба бюджета непротиворечивы только если 22 GB — первый этап 80-гигабайтного
   прогона.
6. **Время — главный блокер, а не GiB.** При измеренной пропускной способности 95M (0.307 с/шаг
   на b8/s1024 = 26.7 kB/s) 1B на 22 GB — это **103-247 суток** непрерывного счёта чистым
   byte-level; на 80 GB — **376-900 суток**. Byte-patch (8 байт на позицию) режет это в
   измеренные **4.6×** (бриф), то есть 22 GB → **22-54 суток**, 80 GB → **82-196 суток** — но
   качество патча ещё решается A/B, и без него счёт в месяцах. Даже при 15.2 TFLOP/s (33 % пика)
   22 GB чистым байтом — 3.4 месяца. Это не «прогон на вечер», это проект на квартал-год.
7. **Три самых дешёвых рычага, по порядку:** (1) `muon_state_dtype=bf16` — **−1.91 GiB**, 0 %
   времени, уже измерен как бесплатный; (2) полный layer-checkpointing — **−2.76 GiB** на
   b1/s2048 и **−88.2 GiB** на b8/s8192, +33 % времени, точен, но для transformer-стека его в
   коде **нет** (есть только в loop-block); (3) host-offload состояния оптимизатора —
   **−3.83 GiB**, точен, 1.25 с/шага по PCIe. 8-bit Muon даёт **−2.86 GiB** с опубликованным
   паритетом до 2.7B, но он lossy и требует реализации.
8. **Что ломается на 1B:** не инициализация (logit внимания растёт как `0.0004·d`, то есть
   0.33 → 0.72 — мягко), а **переносимость LR**: схема не μP, поэтому LR с 95M не обязан
   переноситься; Muon-шаг ∝ `lr/√cols`, то есть на 1.47× меньше при том же lr; и
   чувствительность к LR растёт с глубиной быстрее, чем с шириной. Всё это проверяется на
   одной карте за 1-3 GPU-часа, а арифметика — за 0.

---

## 0. Что именно посчитано (и что в этом счёте измерено)

| quantity | value | source |
|---|---:|---|
| parameters | 1,028,724,480 | [C] 28×(4·1792² + 2·1792·6656 + 4·1792) + 2·256·1792 + 1792, cross-checked with `param_count` on meta |
| Muon group | 1,027,604,480 | [M] `is_muon_param`, `train/optim.py:1667-1689` |
| AdamW group | 1,120,000 | [M] embed + head + LayerNorms |
| fp32 master weights | **0 GiB** | [M] there are none: «the parameters themselves are fp32», `scripts/vram_ledger.py:287-288` |
| 45.7 TFLOP/s bf16 | the card's peak | [M] `scripts/vram_ledger.py:48` |
| 1.0 GiB | CUDA context + allocator | [M] `port/src/dormouse/model/act_mem.py:4-5` |
| 34 B/token/hidden-unit/layer | activations without recompute, bf16 + FlashAttention | [T] `2205.05198v1` Table 2 (`sbh(34)`) + Eq. 6; restated in `docs/research/2026-11-08-activation-memory.md:139` |

**Почему 34 B, а не наш якорь 300 KiB/токен.** Наш собственный якорь даёт
300 KiB/токен при L=12,d=832 = **30.8 B**/токен/единицу/слой
(`docs/research/2026-11-08-activation-memory.md:132`), то есть на 10.5 % *ниже* мегатроновских
34 B. Разница объяснена в том же файле (:229): у нас нет dropout-масок и fused RMSNorm. Я беру
**34 B как консервативную границу** и помечаю все активационные числа как ±15 %.

**Формулы (все [C]):**

```
params   = 4·N                                  (fp32)
grads    = 4·N                                  (fp32)
muon     = 4·N_muon   |  2·N_muon (bf16)  |  1.0156·N_muon (8-bit, fp32 scale / 256)
adamw    = 8·N_adamw                            (exp_avg + exp_avg_sq, fp32)
NS       = 6·min(N_muon, muon_chunk_elems)·4    (muon_chunk_elems default 1<<25)
ctx      = 1.0 GiB
act_nockpt = 34·L·d·b·s
act_ckpt   = (4·L + 34)·d·b·s                   (4 B = fp32 residual input per layer)
act       += 4·b·s·V                            (logits at the head, fp32)
```

`muon_chunk_elems=1<<25` — дефолт `TrainConfig` (`port/src/dormouse/config.py:2030`); при
`1<<26` NS-воркспейс удваивается до 1.5 GiB и это самая дешёвая обратная операция в таблице.

---

## 1. Памятный ledger для 1B на 16 GiB

Все числа — **GiB** (2³⁰). `NS` — пик Muon-Newton-Schulz, `ctx` — контекст/аллокатор.
`fits` — против рабочего потолка **15.0 GiB** (`vram_ledger.py:50`), в скобках — против
физических 16.0.

| b | s | ckpt | Muon | seg | params | grads | muon | adamw | act | NS | ctx | **TOTAL** | fits |
|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 2048 | – | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 3.26 | 0.75 | 1.00 | **16.51** | ✗ |
| 1 | 2048 | – | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 3.26 | 0.75 | 1.00 | **14.59** | ✓ (0.41) |
| 1 | 2048 | ✓ | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 0.50 | 0.75 | 1.00 | **13.75** | ✓ |
| 1 | 2048 | ✓ | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 0.50 | 0.75 | 1.00 | **11.84** | ✓ |
| 8 | 2048 | – | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 26.05 | 0.75 | 1.00 | **39.30** | ✗ |
| 8 | 2048 | – | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 26.05 | 0.75 | 1.00 | **37.38** | ✗ |
| 8 | 2048 | ✓ | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 4.01 | 0.75 | 1.00 | **17.26** | ✗ |
| 8 | 2048 | ✓ | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 4.01 | 0.75 | 1.00 | **15.34** | ~ (0.66) |
| 1 | 8192 | – | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 13.02 | 0.75 | 1.00 | **26.27** | ✗ |
| 1 | 8192 | – | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 13.02 | 0.75 | 1.00 | **24.36** | ✗ |
| 1 | 8192 | ✓ | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 2.00 | 0.75 | 1.00 | **15.25** | ~ (0.75) |
| 1 | 8192 | ✓ | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 2.00 | 0.75 | 1.00 | **13.34** | ✓ |
| 8 | 8192 | – | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 104.19 | 0.75 | 1.00 | **117.44** | ✗ |
| 8 | 8192 | ✓ | fp32 | – | 3.83 | 3.83 | 3.83 | 0.008 | 16.03 | 0.75 | 1.00 | **29.28** | ✗ |
| 8 | 8192 | ✓ | bf16 | – | 3.83 | 3.83 | 1.91 | 0.008 | 16.03 | 0.75 | 1.00 | **27.37** | ✗ |
| 1 | 8192 | ✓ | fp32 | 2048 | 3.83 | 3.83 | 3.83 | 0.008 | 0.51 | 0.75 | 1.00 | **13.76** | ✓ |
| 1 | 8192 | ✓ | bf16 | 2048 | 3.83 | 3.83 | 1.91 | 0.008 | 0.51 | 0.75 | 1.00 | **11.84** | ✓ |
| 1 | 32768 | ✓ | fp32 | 2048 | 3.83 | 3.83 | 3.83 | 0.008 | 0.53 | 0.75 | 1.00 | **13.78** | ✓* |
| 1 | 32768 | ✓ | bf16 | 2048 | 3.83 | 3.83 | 1.91 | 0.008 | 0.53 | 0.75 | 1.00 | **11.87** | ✓* |
| 8 | 2048 | ✓ | 8-bit | – | 3.83 | 3.83 | 0.97 | 0.008 | 4.01 | 0.75 | 1.00 | **14.40** | ✓ |
| 8 | 2048 | ✓ | offload | – | 3.83 | 3.83 | 0.00 | 0.008 | 4.01 | 0.00 | 1.00 | **12.68** | ✓ |

`*` — сегментные строки **точны только для KDA/CSA**; для `block_form="transformer"`
`segment_len` отказан (`config.py:5284-5290`), и та же таблица описывает *оконную* модель, а не
нашу (см. §1.3).

### 1.1 Кто влезает

| configuration | verdict | why |
|---|---|---|
| b1/s2048, no ckpt, fp32 Muon | **no** | 16.51 GiB > 16.0 physical |
| b1/s2048, no ckpt, bf16 Muon | **yes, no headroom** | 14.59; 0.4 GiB will not survive allocator fragmentation |
| b1/s2048 + ckpt, fp32 | yes | 13.75 |
| b1/s2048 + ckpt, bf16 | **yes, comfortable** | 11.84; the recommended entry point |
| b1/s8192 + ckpt, bf16 | yes | 13.34 |
| b1/s8192 + ckpt, fp32 | borderline | 15.25 against a 15.0 ceiling |
| b8/s2048 + ckpt, bf16 | **borderline / no** | 15.34 at 15.0; formally inside 16.0, practically not |
| b8/s2048 + ckpt, bf16 + 8-bit Muon | yes | 14.40 |
| b8/s2048 + ckpt, Muon offload | yes | 12.68 (+1.25 s/step) |
| b8/s8192, any combination | **no** | minimum 27.37 |

**Практический вывод.** Единственная конфигурация, которую я бы назвал безопасной на этой
карте, — **b1/s2048, полный checkpointing, bf16 Muon: 11.84 GiB**. Всё остальное — либо
впритык, либо требует нового кода (offload/8-bit), либо меняет модель (сегменты).

### 1.2 Почему b=8 — не самоцель

`grad_accum` не меняет пик памяти (`vram_ledger.py:772-780`: «free VRAM: peak is one
micro-batch; same total FLOPs»). b1×accum8 и b8×accum1 видят один и тот же поток байт и делают
одинаковое число optimizer-шагов. Разница только в загрузке GEMM: при b=1/s=2048 матрица
`[2048,1792]×[1792,1792]` всё ещё достаточно велика, чтобы тензорные ядра не простаивали.
Так что «1B влезает только на b=1» стоит читать как «1B влезает, если не жадничать с
микро-батчем».

### 1.3 Сегменты: что именно ломается

| claim | evidence |
|---|---|
| the `segment_len` carry contract exists and is exact for KDA/CSA | `docs/reviews/2026-11-08-segment-csa.md` §1-§4: the chunk phase is carried, `pos = (count + arange(t)) // chunk + n`; parity Δlogits 1.25e-6…1.91e-6 on the CPU fixture |
| for full attention a segment cut is a different model | [M] `config.py:5284-5290`: «the segmented path carries the KDA recurrence's state across cuts, and the vanilla decoder has no state to carry (its K/V context is the window)» |
| exact 32K at 1B needs a KV cache | [C] 2·L·s·d·2 B = 2·28·32768·1792·2 = **6.13 GiB** (MHA, bf16); GQA-4 — 0.88 GiB |
| exact 1M at 1B is impossible in principle | [C] the same arithmetic at s=1e6: **187 GiB** of KV alone |
| KDA state at 1M — 393 KB, CSA fine 32 MB, HCA 1 MB ≈ 34 MB | [M] brief; at 8K/32K this is noise (<0.1 % of the budget) |

То есть «сегментированное обучение на 8K/32K» — это **выбор архитектуры**, а не флаг памяти.
Если 1B остаётся vanilla-transformer, единственный честный способ увидеть 32K — оконное
внимание (и тогда 32K-качество надо мерить отдельно), либо GQA + KV-carry (0.88 GiB, но это
уже другая модель со своей геометрией).

---

## 2. Что реально обучено на ~1B на одной карте 16-24 GiB

**Короткий ответ: ничего.** Ни одной статьи и ни одного заслуживающего доверия блога, где
~1B обучают с нуля на одной карте ≤24 GiB, я не нашёл. Поиск: arXiv API по
`"single GPU" AND training AND billion`, `"consumer GPU" AND language model AND training`,
`"1B" AND pretraining AND "single GPU"`, `"Muon" AND quantized`, плюс собственный
литературный обзор репозитория (`docs/research/2026-10-15-1b-tech-verification.md`). Всё, что
находится, — это либо fine-tuning/LoRA на одной карте, либо pretrain на 8-64 картах.

| run | params | tokens | hardware | GPU-hours | on our card |
|---|---:|---:|---|---:|---:|
| Pythia-1.0B [T] `2304.01373v2` Table 3 | 1.0B | 300B | **64×A100-40G** | 4,830 [T] `2401.02385v2` Table 2 | **9.0 years** [C] |
| TinyLlama-1.1B [T] `2401.02385v2` §2 | 1.1B | 3T (v1.1: 2T) | **16×A100-40G** (v1.1: 16 nodes × 4) | 34,560 [C] from 3,456/300B [T] Table 2 | **41-99 years** [C] |
| Cramming [T] `2212.14034v1` | masked-LM, GLUE; below 1B [C] | — | **1 consumer GPU, 24 h** | 24 | a day |

Вне таблицы, потому что это speedrun, а не pretrain: nanochat (2026-01) доводит модель
depth-24 до GPT-2 CORE за **~2 ч на 8×H100, ~$48** [web] `karpathy/nanochat` README. По FLOP
это 2.3e19 при 40 % MFU → **3-6 недель** на нашей карте [C,S].

**Пересчёт [C].** FLOPs = 6·N·D. Pythia: 6·1.0e9·3e11 = 1.8e21 → /6.35e12 = 2.84e8 с = 9.0 лет.
TinyLlama: 6·1.1e9·3e12 = 1.98e22 → 3.12e9 с = 99 лет (при 15.2 TFLOP/s — 41 год). Для
nanochat: 16 H100-ч при 40 % MFU ≈ 2.3e19 FLOP → /6.35e12 = 3.6e6 с = 42 суток; при вдвое
меньшем MFU — 21 сутки. **Это диапазон, а не число:** [S], потому что MFU H100 в блоге не
опубликован.

| conclusion | evidence |
|---|---|
| at 1B the bottleneck is throughput, not memory | 1B fits at b1 (§1), but 22 GB is 103-247 days of pure byte (22-54 with patches, §3.3) |
| the closest published point is 16×A100-40G | [T] TinyLlama §2: «We pretrain TinyLlama with 16 A100-40G GPUs» |
| the hardware gap is 64 cards (Pythia) and 16 cards (TinyLlama) against one | [T] the tables above |

**Честная формулировка для владельца.** Опубликованного 1B-на-одной-карте нет не потому, что
это невозможно по памяти (влезает), а потому, что на одной карте это годы счёта на один прогон.
Если такой прогон и существует, он не индексируется теми терминами, которые я проверил.

---

## 3. Chinchilla для наших двух бюджетов

Считаем в BPE-эквивалентах: **1 BPE-токен ≈ 4 байта** (наша byte-level модель видит 4× больше
позиций на тот же текст — это отдельная проблема, см. §3.3).

| budget | bytes | BPE tokens | N at 20 tok/param | D/N at N=1B | compute at N=1B |
|---|---:|---:|---:|---:|---:|
| 22 GB | 22e9 | 5.5e9 | **275M** | 5.5 | 1.36e20 FLOP |
| 80 GB | 80e9 | 20e9 | **1.00B** | 20.0 | 4.94e20 FLOP |

**Два разных правила, и их нельзя путать [C]:**

* **Правило данных** (D = 20·N): 22 GB → N = 275M; 80 GB → N = 1.00B. Это то, что обычно
  называют «Chinchilla-точкой» для бюджета данных.
* **Правило вычислений** (C = 6ND, минимизируем loss по N при фиксированном C):
  N\* = √(C/120). Для C = 1.36e20 (наш прогон 1B×22 GB) → **N\* = 1.06B, D\* = 21.3e9
  BPE-токенов = 85 GB**. Для C = 4.94e20 (1B×80 GB) → N\* = 2.03B, D\* = 40.6e9 = 162 GB.

**Что это значит.** План «1B на 22 GB» тратит вычисления, достаточные для 1B на **85 GB**, но
берёт только 22 GB. Это не «недостаточно параметров» — это **в 4 раза недостаточно данных для
своего же compute-счёта**. Наоборот, «1B на 80 GB» — единственная точка, где 1B и 20 ток/пар
совпадают точно. Если 22 GB — самостоятельный финальный прогон, то по обоим правилам
оптимальный размер ~275M, а не 1B.

### 3.1 Overtraining маленьких моделей: что говорит литература

| claim | source | number |
|---|---|---|
| quality keeps improving to 10,000 tokens/parameter; no «saturation point» found | [T] `2401.00448v3` §4 | 47 models 150M-6B, 10-10,000 tok/param |
| under large inference demand it is optimal to train **smaller and longer** than Chinchilla | [T] `2401.00448v3` abstract | ~10⁹ requests |
| 4 epochs of repeated data is within noise of unique data | [T] `2305.16264v5` abstract | 900B tokens, 9B models (the top of the tested range) |
| Ripple-65B: 65B × 12T = 185 tok/param, $32M, 40 % MFU | [ours/T] `docs/reviews/2026-11-08-ripple-65b.md:16-30` | 9.25× Chinchilla |
| TinyLlama-1.1B: 1.1B × 2-3T = 1800-2700 tok/param | [T] `2401.02385v2` abstract | 90-135× Chinchilla |

**Вердикт.** Вся литература про overtraining — про движение **выше** 20 ток/параметр
(185, 1800, 10 000): она говорит, что вправо от 20 можно идти долго, а не что можно стартовать
слева. 1B на 22 GB — это 5.5 ток/пар; при **фиксированных** данных больше параметров всё равно
лучше по loss (§3.2), но за это платится 4× вычислений. То есть выбор 1B на 22 GB — это выбор
заплатить 4× за ~0.2 ната, а не «дешёвый overtraining». Ripple с его 185 ток/пар доказывает,
что overtraining работает; для наших 22 GB он не аргумент, потому что мы там ниже 20, а не выше.

### 3.2 Что будет с loss (грубо, [S])

По фиту Chinchilla (`L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28`, `2203.15556v1` Table A3) при
D = 5.5e9: N=1B → 2.81; N=275M → 3.01. То есть при **фиксированном** D больше параметров
*лучше*, и «275M оптимален» — это про оптимальность по вычислениям, а не по loss при
фиксированных данных. Плата за 1B при D=5.5e9 — 4× вычислений за ~0.2 ната. Формула
экстраполирована ниже своей калибровки, поэтому это [S], а не прогноз.

### 3.3 Байтовая модель платит 4× за токен, а патч возвращает 4.6×

Один и тот же текст в byte-level — это 4× больше позиций, чем в BPE, и каждая позиция стоит
6N FLOP. Значит при равном тексте наша byte-модель делает **4× больше вычислений**, чем
BPE-модель того же N [C]. Byte-patch (8 байт на позицию) делит это на 8 теоретически и на
**4.6** измеренно (b=32, бриф), то есть патч-модель на байт **дешевле** BPE-модели того же N
примерно вдвое.

**Важно: это меняет время, а не бюджет данных.** 22 GB остаются 22 GB = 5.5B BPE-эквивалентов,
и вывод §3 про Chinchilla от патча не зависит. Меняется счёт: 22 GB чистым байтом при 1B —
103-247 суток, с патчем — **22-54 суток** [C] (4.6× из брифа); 80 GB — 376-900 → **82-196**.
Плюс к этому при фиксированном `s` патч-окно покрывает 8× больше байт за тот же пик памяти
(измерено −3.8× VRAM на байт), то есть память на байт тоже падает. Это единственный рычаг в
уравнении, который меняет порядок величины счёта, а не проценты; и он единственный, чьё
качество ещё не решено (A/B идёт прямо сейчас).

---

## 4. Три самых дешёвых рычага, чтобы 1B влез

Ранжировано по «GiB за единицу боли». Все saving'и — разности ledger'а §1, не оценки.

| # | lever | GiB (b1/s2048) | GiB (b8/s8192) | time | exact/lossy | published evidence |
|---|---|---:|---:|---|---|---|
| 1 | `muon_state_dtype=bf16` | **−1.91** | −1.91 | 0 % | store rounds, accumulation stays fp32; 1e-2 relative gate — **lossy but cheap** | [M] brief: «free (measured)»; [M] `vram_ledger.py:291`; [T] `2509.23106v3` §4.4 — 8-bit, not bf16 |
| 2 | full layer checkpointing | **−2.76** | **−88.16** | +33 % (3 fwd → 4) | **exact**, bit-identical on CPU, GPU gate 1e-6 | [T] `1604.06174v2` §3 (O(√n), +1 forward); [T] `2205.05198v1` Table 4: full recompute **+39 %**, selective **+7 %** (22B, 1 layer) |
| 3 | host offload of optimizer state | **−3.83** | −3.83 | +1.25 s/step (2×3.83 GiB / 6.5 GB/s PCIe) | **exact** (same values, moved) | [T] `2101.06840v1` (ZeRO-Offload: 13B on one GPU); [T] `2104.07857v1` (NVMe); [M] `vram_ledger.py:752-766` — priced for this card |
| 4 (bonus) | 8-bit Muon state | **−2.86** | −2.86 | 0 % | **lossy**, parity on loss/benchmarks | [T] `2509.23106v3` abstract: parity, **62 %** state reduction, tested at 2.7B; [T] `2605.11396v2` MuonQ — a low-bit variant with directional fidelity |
| 5 (not a lever) | 8-bit AdamW (`optim_8bit`) | **−0.000** | −0.000 | 0 % | — | [M] the AdamW group is 1.12M params = 0.008 GiB; nothing to save (confirms the brief) |

**Порядок применения, если цель — «влезть и не потерять качество»:**
1. `muon_state_dtype=bf16` — бесплатно, делать сразу, без вариантов. Это не тюнинг, это
   обязательный шаг: без него b1/s2048 без ckpt не влезает вообще.
2. **offload состояния** — самый большой точный выигрыш (−3.83) и самая дешёвая реализация:
   в репозитории уже есть pinned-host пул и политика бюджета для активаций
   (`port/src/dormouse/model/act_mem.py`), их надо переиспользовать, а не писать заново.
   Цена 1.25 с/шага при шаге 1B в 2-16 с — это 8-60 %, надо мерить.
3. **checkpointing** — самый большой выигрыш на длинном контексте (−11 GiB при b8/s8192), но
   для `block_form="transformer"` его **нет в коде**: `activation_recompute` живёт только в
   loop-block (`docs/research/2026-11-08-activation-memory.md` §0, строка `iter_checkpoint`).
   Это не флаг, это работа: обернуть `TransformerBlock` в `checkpoint(use_reentrant=False)`.
   Пока её нет, «−79.9 %» из брифа к нашему стеку не относится — лога в репозитории нет, и
   §9 того же документа это признаёт.

**Чего в этом списке нет и почему.** Сегментирование с переносом состояния — сильнейший рычаг
по асимптотике (s → C), но он существует только для KDA/CSA (§1.3) и потому не является
рычагом для текущей 1B-формы. Шардирование состояния (ZeRO-1) требует второй карты; на одной
карте это не применимо.

---

## 5. Что ломается на 1B и не ломается на 95M

| mechanism | what changes at 832 → 1792 / 12 → 28 | evidence | cost to test on our card |
|---|---|---|---|
| **initialization** | `std=0.02` is fixed (`model/transformer.py:192-196`), the input is LayerNorm'd → `σ(q)=0.02·√d`: 0.577 → 0.847, and `std(logit)=0.0004·d`: **0.33 → 0.72** (×2.15). This is **mild** — no saturation at init; but a fixed std means the function changes with width, so the 95M LR does not transfer by construction | [C] from the code; [T] `2309.14322v2` §3.1.1 — the logit-growth mechanism exists and is reproducible at small scale, but it does not fire here at d=1792 | **0 GPU-minutes**: one forward on random data at d=832 and d=1792, measure logit std and attention entropy (a confirmation, not a search) |
| **LR** | η\* is a power law in N and D; B\* depends on D only | [T] `2503.04715v7` abstract (3,700 models, 100T tokens); [T] `2609.27581v1` abstract: a refit below 59M gives η\* = 0.0985·N^-0.508·D^0.238, B\* = 3.6e-4·D^0.931 | 200 steps × 2 LRs = **2 GPU-hours** |
| **Muon scaling across shapes** | `RMS(ΔW) = lr·√(max(1,r/c))/√c` — a constant of the SHAPE, not of the gradient. At lr=3e-4: 1.04e-5 on [832,832] → **7.09e-6** on [1792,1792], i.e. **×1.47 smaller** at the same lr; on [1792,6656] — 3.68e-6 | [M] `docs/reviews/2026-11-07-muon-audit.md:24,31,113-125` | **0 GPU-minutes**: an update-RMS table over every 1B shape from the same formula |
| **gradient clipping** | clipping is global, pre-step, scale-invariant; at depth 28 the LR sensitivity grows **faster than with width** | [M] `docs/reviews/2026-11-07-optimizer-suite.md`; [T] `2309.14322v2` §3.2.3, Figure 7 | included in the same 200 steps |
| **μP / hyperparameter transfer** | our code is **not μP**: a fixed std and one scalar lr. μP wants 1/√fan_in and lr ∝ 1/width for embed/head; depth-μP wants 1/√depth on the residual branch — we have only the second half (`0.02/√(2L)`) | [T] `2203.03466v2` (μTransfer); [T] `2309.16620v2` (depth-μP); [T] `2310.02244v5` (TP VI: for deep transformer blocks every infinite-depth limit has fundamental limitations) | 2 arms × 200 steps at 95M and 1B = **4 GPU-hours** |
| **depth/width aspect** | 1792/28 = 64 — the same shape as 832/12 = 69 and Ripple's 5120/80 = 64. «DeepNarrow» is Pareto-better at fixed parameters | [T] `2109.10686v2` §4, Table 4: T5 Small 16L ≈ Base, **−40 % time, −50 % parameters**; [T] `2309.14322v2` — the price of depth is LR sensitivity | 0 (a config choice) |
| **exact long context** | 32K KV = 6.13 GiB (MHA); 1M = 187 GiB; a segment without carry is a different model | [C] §1.3; [M] `config.py:5284-5290` | 2 arms × 5 GPU-hours (2K window vs 8K window, equal bytes) |

**Главное.** Три из семи пунктов (инициализация, Muon-формы, аспект) проверяются **без GPU** —
это арифметика и один CPU-forward. Два (LR, клип) — 200 шагов. Два (μP, сегменты) — реальные
A/B на 4-10 GPU-часов. Ни один не требует 1B-прогона целиком, и это единственная хорошая
новость в этом отчёте.

### 5.1 Про инициализацию — арифметика говорит «не она»

При `std=0.02` и LayerNorm'енном входе `σ(q) = 0.02·√d` (0.577 при d=832, 0.847 при 1792), а
логит `= q·k/√hd` по `hd`-мерной голове имеет `std = 0.0004·d` — **0.333 при 832 и 0.717 при
1792** [C]. Softmax с σ=0.72 — это обычный, не saturation режим: init на этом масштабе **не
ломается**.

**Что действительно ломается — это переносимость LR.** Фиксированный `std=0.02` вместо
μP-шного `1/√fan_in` означает, что при росте ширины меняется сама функция (выход слоя растёт
как √d), поэтому оптимум LR, найденный на 95M, не обязан быть оптимумом на 1B. Это ровно то,
что `2203.03466v2` (μTransfer) обещает исправить, и чего у нас нет. При d=1792 эффект мягкий
(σ(q) 0.58 → 0.85), но он есть, и он складывается с Muon-фактором §5 (шаг ×1/1.47).

**Проверка за 1 минуту на CPU:** forward на random data при d=832 и d=1792, замерить std
логитов и энтропию внимания. Ожидание: 0.33/0.72 и почти одинаковая энтропия — то есть
подтверждение «init не виноват», а не поиск проблемы. Реальная проверка — LR A/B (2 GPU-часа).

---

## 6. Самое дешёвое измерение на каждое «не знаю»

| question | what to measure | cost |
|---|---|---|
| does 1B fit in practice | 200 steps at b1/s2048, ckpt, bf16 Muon; `max_memory_allocated` + s/step | **~1 GPU-hour** |
| is the init to blame at 1792 | CPU forward at d=832 vs d=1792, attention logit std and entropy (expected: no) | **1 minute, 0 GPU** |
| is the Muon lr right at 1B | an `RMS(ΔW)` table over the 1B shapes from `muon-audit.md`; then 2 LRs × 200 steps | 0 + 2 GPU-hours |
| is a 2K window worth an 8K window | 2 arms × 5 GPU-hours, equal bytes, BPB | 10 GPU-hours |
| the real 1B throughput | the same 200-step run: bytes/s → days for 22/80 GB | included in the 1 GPU-hour |

---

## 7. VERIFIED (sourced) — что можно утверждать

1. **N = 1,028,724,480** для L=28/d=1792/f=6656/vocab 256; Muon 1,027,604,480, AdamW 1,120,000
   [M] meta-инстанцирование + `is_muon_param`.
2. **Статическое состояние 1B = 13.25 GiB (fp32 Muon) / 11.34 GiB (bf16 Muon)**, включая
   1.0 GiB контекста и 0.75 GiB NS [C] из `vram_ledger.py:279-300` + `config.py:2030`.
3. **b1/s2048 + ckpt + bf16 Muon = 11.84 GiB**; b8/s8192 = 27.37 GiB при тех же рычагах [C].
4. **b8 при 1B не влезает**: лучшая конфигурация без нового кода — 15.34 GiB при потолке 15.0 [C].
5. **8-bit AdamW не экономит ничего** (0.008 GiB), потому что AdamW-группа — 1.12M параметров
   [M].
6. **bf16 Muon экономит 1.91 GiB и не стоит времени** [M] бриф + `vram_ledger.py:291`.
7. **Полный checkpointing: +33 % времени, −2.76 GiB на b1/s2048, −88.16 GiB на b8/s8192** [C];
   в литературе +39 % у Мегатрона на 22B [T] `2205.05198v1` Table 4.
8. **Offload состояния: −3.83 GiB, 1.25 с/шага** [M/C] `vram_ledger.py:752-766`.
9. **8-bit Muon: 62 % сокращение state, паритет до 2.7B** [T] `2509.23106v3` abstract.
10. **`segment_len` для vanilla transformer отказан кодом** [M] `config.py:5284-5290`; контракт
    carry существует и точен для KDA/CSA [M] `docs/reviews/2026-11-08-segment-csa.md`.
11. **Pythia-1.0B: 64×A100-40G, 4 830 A100-ч; TinyLlama-1.1B: 16×A100-40G, ≈34 560 A100-ч**
    [T] `2304.01373v2` Table 3, `2401.02385v2` §2 + Table 2.
12. **20 ток/пар: 22 GB → 275M, 80 GB → 1.00B** [C] из 4 байта/токен.
13. **Качество растёт до 10 000 ток/пар; inference-спрос сдвигает оптимум к меньшим моделям**
    [T] `2401.00448v3` abstract + §4.
14. **≤4 эпох повторных данных ≈ уникальные** [T] `2305.16264v5` abstract.
15. **LR-чувствительность растёт с глубиной быстрее, чем с шириной** [T] `2309.14322v2` §3.2.3.
16. **Muon-шаг ∝ lr·√(max(1,r/c))/√c, константа формы** [M] `muon-audit.md:24,31`.
17. **DeepNarrow Pareto-лучше при фиксированных параметрах** [T] `2109.10686v2` Table 4.

## 8. SPECULATION (my inference) — что нельзя утверждать

1. **[S] «1B на 22 GB — это 4× недостаточно данных для своего compute»** — вывод из
   Chinchilla-фита `2203.15556v1`, экстраполированного ниже калибровки; сам фит на таких
   отношениях D/N не проверялся.
2. **[S] «При 1B первым сломается LR, а не init»** — арифметика init [C] показывает мягкий
   рост (0.33→0.72), но реальный эффект отсутствия μP на 1B не измерен; это ожидание, а не
   факт. Обратное тоже возможно: на 28 слоях LR-чувствительность вырастет раньше, чем на
   ширине.
3. **[S] «275M — правильный размер для 22 GB»** — верно только если цель — минимум loss при
   фиксированных вычислениях. Если цель — «модель, которую потом доучим на 80 GB», то 1B на
   22 GB — нормальный первый этап, и это решение владельца, а не арифметики.
4. **[S] «b1×accum8 эквивалентен b8»** — по памяти и по числу optimizer-шагов да; по скорости
   GEMM на b=1 может быть на 10-30 % медленнее, это не измерено на 1B.
5. **[S] «nanochat-прогон на нашей карте = 3-6 недель»** — пересчёт по FLOP при неизвестном
   MFU H100; погрешность до 2×.
6. **[S] «1B на одной карте нигде не опубликован»** — это отрицательный результат поиска, а не
   доказательство несуществования.

**Confidence.** Памятный ledger: **высокая** для статических членов (±2 %, точные счётчики и
dtypes), **средняя** для активаций (±15 %, 34 B vs наши 30.8 B). Вердикт «b8 не влезает»:
**высокая**. Оценки времени: **низкая** — два числа в брифе расходятся в 2.4 раза (0.307 с/шаг
⇒ 15.2 TFLOP/s = 33 % пика, а не 13.9 %; строка `docs/reviews/2026-11-07-night-architecture.md:87`
даёт 52.3 kB/s и 13.9 % одновременно, что с 6N не сходится — 52.3 kB/s при 95M требуют
29.8 TFLOP/s). Пока это расхождение не измерено заново, все «дни» в этом отчёте — диапазон,
а не число. Литературные утверждения: **высокая** (первичные источники, id + версия).

---

## 9. Источники

**Наши (repo):**
`scripts/vram_ledger.py:48,50,279-300,752-766` · `port/src/dormouse/model/act_mem.py:4-5` ·
`port/src/dormouse/model/transformer.py:34-40,192-196` ·
`port/src/dormouse/train/optim.py:1667-1689` · `port/src/dormouse/config.py:2030,5284-5290,4851` ·
`port/configs/flagship3_dense.toml` · `docs/reviews/2026-11-08-segment-csa.md` ·
`docs/reviews/2026-11-07-muon-audit.md:24,31,113-125` ·
`docs/reviews/2026-11-08-ripple-65b.md:16-30` ·
`docs/research/2026-11-08-activation-memory.md:12,132-139,163-165,229` ·
`docs/reviews/2026-11-07-night-architecture.md:87` · `docs/research/2026-10-15-1b-tech-verification.md`

**Литература:**
`2203.15556v1` Chinchilla (Table A3) · `2401.00448v3` Beyond Chinchilla-Optimal ·
`2305.16264v5` Scaling Data-Constrained LMs · `2502.16982v1` Muon is Scalable ·
`2509.23106v3` 8-bit Muon · `2605.11396v2` MuonQ · `2110.02861v2` 8-bit Optimizers ·
`2101.06840v1` ZeRO-Offload · `2104.07857v1` ZeRO-Infinity · `1604.06174v2` Sublinear Memory ·
`2205.05198v1` Reducing Activation Recomputation (Table 2, Table 4) ·
`2203.03466v2` Tensor Programs V (μP) · `2309.16620v2` Depthwise HP Transfer ·
`2310.02244v5` Tensor Programs VI · `2309.14322v2` Small-scale proxies (Figure 1, §3.2.3) ·
`2503.04715v7` Step Law · `2609.27581v1` Step Law below 59M · `2109.10686v2` Scale Efficiently
(Table 4) · `2304.01373v2` Pythia (Table 3) · `2401.02385v2` TinyLlama (§2, Table 2) ·
`2212.14034v1` Cramming · [web] `karpathy/nanochat` README (2026-01).

## 10. Открытые вопросы

1. **Расхождение MFU.** 0.307 с/шаг vs 13.9 % MFU несовместимы при 6N. Нужен один прогон с
   `max_memory_allocated` и FLOP-счётчиком, иначе все сроки — вилка 2.4×.
2. **Активационная константа.** 34 B (Megatron) против 30.8 B (наш якорь) на нашем стеке не
   различались measurement'ом; 10 % разницы — это 1.2 GiB на b1/s8192.
3. **Checkpointing для `TransformerBlock` не реализован.** «−79.9 %» из брифа к нему не
   относится; нужен порт `checkpoint()` в `model/transformer.py` и GPU-гейт.
4. **8-bit Muon на нашей карте не проверен.** Паритет опубликован до 2.7B, но у нас sm_120 и
   свой формат состояния; цена — один A/B на 95M.
5. **Сегментный carry для full attention не существует** ни в коде, ни в литературе как
   «точный» приём; если 32K нужен на transformer-форме, это GQA+KV-carry или оконная модель,
   и качество надо мерить.
6. **Не проверено:** влезает ли b8/s2048 с bf16 Muon+ckpt на реальном аллокаторе (15.34 GiB
   при 16.0 физических — это 0.66 GiB запаса, чего обычно не хватает).
