# 2026-11-08 — kernel audit: where the next 10–20% of the `flagship3_dense` step is

Read-only audit of the live lane `port/`. **CPU only** — the card is held by the
`csa8k` run, so every number here is either read out of a recorded artifact, or
measured on the CPU with the repo's own instruments (dispatch census, launch
count, FLOP counter). **No GPU work was run; nothing was committed.**

**Line numbers are `HEAD = 570546b`** (`git show HEAD:<path>`). The worktree is
dirty — another lane has +367 uncommitted lines in
`port/src/dormouse/train/loop.py` and +25 in `config.py` — so a *worktree* line
number is not reproducible and is not used. HEAD has since moved to `3072c2a`;
of the files cited here only `scripts/kernel_profile.py` changed, and only below
line 850, so the anchors below still hold.

Marks: **[M]** measured, artifact named · **[C]** computed from a named code
path or from measured inputs · **[S]** inference, with the check that settles it.
A number without an artifact is a defect.

---

## 0. Verdict (RU)

**Шага на 10–20% одним флагом больше нет. Всё, что осталось, — либо новый
kernel, либо пропускная способность, а не время шага.**

1. **145.8 мс шага (56.2 кБ/с) — это НЕ файл рецепта.** `port/configs/flagship3_dense.toml`
   резолвится в `seq_len=512`, `grad_accum=1`, `ns_dtype=fp32`, `ns_steps=5`,
   `muon_state_dtype=fp32`, `muon_lr_mult=1.0`, `ckpt_every=100` — измеренный
   аналог этого прогона 283–290 мс/шаг, ~28 кБ/с. 56 кБ/с существуют только как
   `TOML + шесть --set` в `scripts/throughput_ladder.sh` / `recipe_ladder.sh` /
   `train_budget.sh`. Три разных программы названы «одним рецептом» (§1).
2. **Шаг делает 7.52 TFLOP арифметики:** 5.66 TFLOP модель + 1.85 TFLOP
   Newton–Schulz (5 шагов). Это **51.5 TFLOP/s = 58%** от паспортных 88.7
   bf16/FP32-accumulate. 42% разрыва — не «медленное ядро», а работа, которую
   можно только *убрать*.
3. **Бюджет сходится из двух измеренных точек:** модель fwd+bwd 88.8 мс
   (`(470.0 − 203.6)/3`, `bench_compile`, g=3) + оптимизатор 57.0 мс = 145.8.
   Внутри оптимизатора NS = 27.3 мс (дельта 5↔3 шагов, две независимые пары
   рук), остальное — **29.7 мс мелких ядер (20% шага)**.
4. **Внутри модели GEMM — 49–52 мс, всё остальное — 36–40 мс (25–27% шага).**
   В этом остатке attention (8–16 мс) и pointwise-шов (20–30 мс): fp32-остаток,
   bf16↔fp32 касты, LN, GELU, RoPE. Трафик шва — **52.5 ГБ/шаг, верхняя
   оценка**. Реально снимается 8–14 мс (5–10%) одним Triton-эпилогом блока.
5. **Attention: арифметика оставляет math-пути слишком мало места — ожидание
   fused, 0%.** Math-SDPA стоит 20–50 мс только на attention, а на весь не-GEMM
   остаток есть 36–40 мс; math влезает только в самом дешёвом углу (20 мс при
   50 TFLOP/s) и тогда шов должен стоить 16 мс, что ниже его же кастов. Одна
   строка на карте это добивает.
6. **Единственный измеренный выигрыш ≥10% — не время шага, а пропускная
   способность:** `grad_accum=3` на быстрой руке даёт расчётные **76 кБ/с
   (+35%)** из двух измеренных рук; это никогда не запускалось.
7. **Мёртвое, с артефактами:** CUDA-графы (0…−4%, три замера), `torch.compile`
   как он включён (`dynamic=True` = 0.49× на этом шейпе), `muon_state_dtype=bf16`
   ради скорости (0.99×), `ns_steps=3` (+0.27 BPB), `ns_kind=triton` (отказ по
   `MAX_RANK=64`), `mm_dtype=fp8` (нет backward), батч, загрузчик, число запусков.
8. **Отдельный дефект: константа полосы в репозитории завышена вдвое.**
   `scripts/pass_census.py:68` считает 1.9 ТБ/с «HBM» — у 5070 Ti **896 ГБ/с**
   GDDR7. Все census-оценки pointwise-проходов в `docs/` посчитаны вдвое
   оптимистичнее реальности.

---

## 1. Три программы, которые называют «рецептом»

**RU: «живой рецепт» — это не файл. Файл — только форма модели; скорость и
оптимизатор живут в `--set` трёх скриптов. Прогон на 40k шагов (артефакт
`/tmp/opencode/flagship3/flagship3.config.toml`, шаг 5000) шёл на **медленной**
руке — 51.2 кБ/с, а не 56.**

| | A: `flagship3_dense.toml` как есть | B: «production» 40k-прогон | C: 56 кБ/с лента (лестница/E1/budget) |
|---|---|---|---|
| seq_len | **512** (дефолт `config.py:1787`) | 1024 | 1024 |
| grad_accum | 1 | **3** | 1 |
| batch | 8 | 8 | 8 |
| `ns_dtype` | **fp32** | fp32 | **bf16** |
| `ns_steps` | 5 | 5 | 5 |
| `muon_state_dtype` | **fp32** | fp32 | **bf16** |
| `muon_lr_mult` | **1.0** | 12.0 | 12.0 |
| mix / doc_masking | — / false | mix_v7_docmask / true | mix_v7* / false |
| `ckpt_every` | **100** (дефолт) | 5000 | 10⁶ |
| измерено | 283–290 мс/шаг при s1024 [M] | **480 мс, 51.2 кБ/с** [M] | **145.8 мс, 56.2 кБ/с** [M] |
| артефакт | `logs/thru/muon_fp32.jsonl` | `logs/flagship3.jsonl` + snapshot | `logs/ladder/control_refetch.jsonl` |

Источники: `[M]` `load_run("configs/flagship3_dense.toml")` → `seq_len=512,
grad_accum=1, ns_steps=5, ns_dtype="fp32", muon_state_dtype="fp32",
muon_lr_mult=1.0` (поля `config.py:1787, 1957, 2020, 2040, 2065`);
`[M]` snapshot прогона B; `[M]` `logs/thru/*.jsonl` и
`logs/ladder/control_refetch.jsonl`. B — единственный «долгий» прогон рецепта:
он дошёл до шага 10 000 и умер в `save_ckpt` (`flagship3.log`, `RuntimeError:
basic_ios::clear` на 762 МБ файла), так что 40 000 шагов на этой форме не
существует ни в одном артефакте.

**Последствия, которые важнее любых ядер:** запуск файла напрямую тренирует
другую модель (Muon-шаг в 12 раз меньше, окно вдвое короче, чекпойнт каждые 100
шагов = +8% времени прогона), а на 40k-прогоне B оптимизатор был **42% шага** —
то есть половина «медленного рецепта» — это fp32 Newton–Schulz, который в
быстрой ленте уже убран.

### 1.1 Измеренные руки (b8×s1024, 40 шагов, последняя строка лога)

| рука | B/s | мс/шаг | × к `muon_fp32` | что изменено |
|---|---:|---:|---:|---|
| `muon_fp32` | 28 259 | 289.9 | 1.00 | контроль: NS fp32 ×5, state fp32 |
| `ns_bf16` | 54 421 | 150.5 | **1.93** | NS bf16, 5 шагов |
| `ns3` | 36 997 | 221.4 | 1.31 | NS fp32, 3 шага |
| `ns_bf16_3` | 58 686 | 139.6 | **2.08** | NS bf16 ×3 |
| `state_bf16` | 28 019 | 292.4 | 0.99 | state bf16, NS fp32 ×5 |
| `muon_fast` | 57 781 | 141.8 | 2.04 | NS bf16 ×3 + state bf16 |
| `compile` | 28 082 | 291.7 | 0.99 | `torch_compile=true` |
| `graph` | 27 897 | 293.7 | 0.99 | `opt_graph=true` |
| `muon_fast_graph` | 57 449 | 142.6 | 2.03 | fast + `opt_graph` |

`[M]` `/home/sehaxe/logs/thru/*.jsonl` (2026-11-08). Те же числа в
`docs/records/architecture/2026-11-08-the-16x.md:29-38` (там контроль 283 мс).
**RU: bf16-NS — единственный настоящий выигрыш (1.93×), и он уже в живом
прогоне; state bf16, compile и граф — нули.**

---

## 2. Что шаг реально запускает (измерено)

`TorchDispatchMode` + `FlopCounterMode` на **meta**-тензорах, форма
`b8×s1024`, HEAD 570546b. Считаются все aten-опы; «launch» = оп минус
метаданные (`view/detach/expand/…`), по правилу
`scripts/bench_muon_launch.py::LaunchCounter`. Attention на meta идёт
**math-backend** (flash на meta не зарегистрирован) — его строки помечены.

| стадия | aten-опов | запусков | трафик (чтение+запись) | что это |
|---|---:|---:|---:|---|
| forward+backward модели | 2 639 | **1 384** | — | 12 блоков × (4 GEMM + SDPA + 2 LN + GELU + RoPE) |
| … `mm` | 147 | 147 | 8.2 ГБ | qkv/proj/fc/fc2, fwd+2×bwd |
| … `bmm` (только math-SDPA) | 72 | 72 | 35.3 ГБ | на карте — flash, этого не будет |
| … `_to_copy` | 330 | 330 | **29.1 ГБ** | bf16↔fp32 касты; вес — только 0.57 ГБ |
| … `add.Tensor` | 157 | 157 | **17.0 ГБ** | fp32-остаток, fwd+bwd |
| … `mul.Tensor` / `mul.Scalar` | 274 | 274 | 5.8 ГБ | RoPE, гейты, масштабы |
| … LN fwd+bwd | 48 | 48 | 3.3 ГБ | `native_layer_norm` + backward |
| … GELU fwd+bwd | 24 | 24 | 3.0 ГБ | `gelu` + `gelu_backward` |
| … `cat` | 36 | 36 | 1.6 ГБ | `apply_rope` |
| оптимизатор | 439 | **193** | ~12 ГБ | momentum + ColRow + writeback + AdamW |
| … `bmm` (NS) | 36 | 36 | 3.3 ГБ | 1 110 GFLOP при 3 шагах (1 852 при 5) |
| firewall + clip + grad_norm | ~120 | ~120 | ~4 ГБ | 117 `nan_to_num_` + 2 foreach |

- `[M]` fwd+bwd: **5 663.7 GFLOP = 691.37 MFLOP/байт** — ровно пин
  `mod_ledger` из шапки рецепта. Число воспроизведено на b=8 независимо.
- `[M]` суммарно ~**1 700 запусков/шаг** (1 384 + 193 + ~120).
- `[M]` pointwise-шов (всё, кроме `mm/bmm/softmax/tril/where`): **52.5 ГБ и
  1 129 запусков**. Это **верхняя оценка**: каждый вход считается по полному
  `numel`, включая виды и повторные чтения, а часть тензоров (27 МБ остатка)
  живёт в L2. **Время из этого не выводится** — и тем более не по 1.9 ТБ/с
  (§6): у карты 896 ГБ/с GDDR7. Настоящая цена шва измеряется как остаток в §3.
- `[M]` host-sync на не-report шаге: **0** — уже измерено на этой же
  архитектуре (`docs/records/reviews/2026-11-08-cpu-perf-audit.md:64-72`: `reads=0
  (device 0)`), и ни один путь шага не содержит `.item()/float()/bool()`;
  единственный `_host_scalar` — на каденции лога (`loop.py:69, 2027`).
- `[M]` лог живого прогона: `forward=7629` (=1/шаг), `transformer_block=91548`
  (=12×), `nan_skipped=0`, `kda=off,n=0`, `engram=0/0` (`flagship3.log`).
  **Счётчиков для fused-AdamW, bf16-NS, SDPA-бэкенда и pointwise-шва нет.**
  В `dict(opt.counters)` после шага — пусто (`[M]`, мой прогон).
- `[C]` MLP — **не GLU**: `fc → F.gelu → fc2` (`model/transformer.py:275-279`),
  гейта и up-пары нет, `ffn_form` этой формой не читается. Фьюзить
  «GLU-эпилог» нечего; единственный эпилог блока — residual add + LN (C1).

**RU: хост не отстаёт — 1 700 запусков при 8–13 мс диспетчеризации утоплены
под 146 мс GPU. Проблема не в запусках, а в трафике, который эти запуски
перекладывают, и в мелких ядрах оптимизатора.**

---

## 3. Бюджет шага (две измеренные точки + арифметика)

Измерено: `bench_compile` на этой же форме, g=3, min-of-reps —
**470.0 мс/шаг, из них `opt.step()` 203.6 мс** (`[M]`
`/home/sehaxe/logs/compile_f3.log`, строки `eager`). Отсюда модель
`(470.0−203.6)/3 = 88.8 мс` на микробатч. Вторая пара — 40-шаговые руки §1.1:
`289.9 − 150.5 = 139.4 мс` — это ровно цена fp32→bf16 в NS, и она сходится с
моделью «NS линеен по шагам» в пределах 3%:

| величина | значение | как получено |
|---|---:|---|
| шаг (лента C) | **145.8 мс** | `[M]` 8192 / 56 181 B/s |
| модель fwd+bwd на микробатч | **88.8 мс** | `[M]` (470.0 − 203.6)/3, g=3 |
| оптимизатор (лента C) | **57.0 мс** | `[C]` 145.8 − 88.8 |
| NS bf16 ×5 | **27.3 мс** | `[C]` из дельты 150.5−139.6 = 10.9 мс на 2 шага |
| NS bf16 ×3 | **16.4 мс** | `[C]` та же дельта |
| NS fp32 ×5 | **171.2 мс** | `[C]` из 289.9−221.4 = 68.5 мс на 2 шага; 171.2 + 29.7 ≈ 203.6 |
| **не-NS оптимизатор** | **29.7 мс (20%)** | `[C]` 57.0 − 27.3 |
| арифметика шага | **7 516 GFLOP** | `[M]` 5 663.7 модель + 1 852 NS (1 110×5/3) |
| темп | **51.5 TFLOP/s** | `[C]` 7 516 / 145.8 мс |
| паспорт карты | **88.7 TFLOP/s** | `[C]` 5070 Ti, bf16/FP32-acc (`docs/records/reviews/2026-11-08-mfu-first-look.md:128-141`) |
| изолированный GEMM | **89–95 TFLOP/s** | `[M]` 0.4398–0.4710 мс на 8192×3072×832 (`compile_f3.log`, autotune) |
| **GEMM-пол модели (без attention)** | **49–52 мс** | `[C]` (5.66−1.0) TFLOP / 89–95 |
| **весь не-GEMM остаток** | **36.5–39.8 мс (25–27%)** | `[C]` 88.8 − 49…52.4 |
| … attention (fused, causal) | 8–16 мс | `[C]` ~0.5 TFLOP при 40–60 TFLOP/s |
| … pointwise-шов | **20–30 мс** | `[C]` остаток после attention |
| … math-SDPA (гипотетически) | 20–50 мс | `[C]` 1.0 TFLOP при 20–50 TFLOP/s |

**Проверка, которая закрывает attention.** Если бы SDPA шёл math-путём, только
attention стоил бы 20–50 мс, а на GEMM уже уходит 49–52 мс — на LN/касты/GELU/
RoPE/CE остаётся от −3 до +10 мс, чего не хватает даже на один слой. Math
влезает только в самом дешёвом углу (20 мс при 50 TFLOP/s), и тогда весь шов
должен стоить 16 мс — меньше, чем одни его касты. Поэтому ожидание — fused-ядро
(8–16 мс), но **это `[S]`, а не измерение**: решает одна строка на карте (§C5).
Колесо под `sm_120` и `flash_attention_*` в `libtorch_cuda.so` — за.

**RU: шаг не «упирается в одно место». Он уже почти равен сумме своих частей:
GEMM на пике, NS на 68 TFLOP/s (77% пика), не-GEMM остаток 36–40 мс — это
двадцать пять процентов, и они трафиковые. 42% разрыва с паспортом — это
20–30 мс шва и 30 мс мелких ядер оптимизатора; убирать надо работу, а не
ускорять её.**

---

## 4. Инвентарь fused-рук: что стоит, что измерено, что здесь недоступно

| рука | в TOML | в ленте C | измерено | артефакт | доступна на этой форме? |
|---|---|---|---|---|---|
| fused residual (`fused_residual`) | выкл | выкл | 729→29 запусков; **−19.5% шага** на mHC-модели | `model/fused_residual.py:24-45` | **НЕТ**: `config.py:4187-4196` отказывает при `use_mhc=false` |
| fused epilogue (`fused_epilogue`) | выкл | выкл | read-цепь parity 4.6e-7…8.5e-7; вся fused-область 729→29 запусков (25×) | `fused_residual.py:977`, `docs/records/reviews/2026-11-08-fused-cluster.md` §2/§5 | **НЕТ**: `config.py:4171-4180`, тот же отказ |
| one-pass combine (`moe_combine="fused"`) | выкл (`torch`) | выкл | один Triton-проход вместо `zeros+index_add_` (три fp32-прохода); счётчик `moe_combine_fused` | `model/moe_kernels.py:1605, 1692` | **НЕТ**: MoE нет (`n_experts=1, moe_topk=0`) |
| fused KDA tail (`fused_kda_tail`) | выкл | выкл | 2 запуска против 11 на хвосте | `model/fused_kda.py:1-40` | **НЕТ**: `use_kda=false` |
| fused AdamW writer (`_foreach_adamw_step`) | **ВКЛ** (дефолт) | **ВКЛ** | 172.6 против 360.0 мс на всей стадии, **но 2.47e-4 BPB расхождения за 20 шагов** | `train/optim.py:2086, 2117`; `compile_f3.log` (`exit 4`) | да — и это баг, не выбор |
| bf16 Newton–Schulz (`ns_dtype=bf16`) | **выкл** | **ВКЛ** | **1.89–1.93× шага**; +0.006 BPB = шум | `docs/records/architecture/2026-11-08-the-16x.md:29-38`; `logs/thru/ns_bf16.jsonl` | да |
| bf16 state (`muon_state_dtype=bf16`) | **выкл** | **ВКЛ** | **0.99×** по скорости; память ×0.5 | `logs/thru/state_bf16.jsonl`; `the-16x.md:31` | да (это memory-рычаг) |
| CUDA-графы (`opt_graph`/`full_graph`) | выкл | выкл | +3%, −4%, 0; 0.99×; 0.99× | `docs/records/reviews/2026-11-05-graph-verdict.md:9-21`; `the-16x.md:36-38` | `full_graph` ещё и отказывает с `doc_masking` (`config.py:4752`) |
| `torch.compile` | выкл | выкл | 1.00× (40 шагов), +2.7% (min-of-5), **0.49× у `dynamic=True`** | `logs/thru/compile.jsonl`; `compile_f3.log` | да, но см. §6 |
| fp8 GEMM (`mm_dtype=fp8`) | выкл | выкл | forward-only, у `_scaled_mm` нет backward | `train/loop.py:1160-1165` | нет для тренировки |

**RU: из всего fused-арсенала на плотной форме доступен ровно один — foreach-AdamW,
и он же единственный, у которого измерено расхождение с torch. Остальные руки
принадлежат mHC/MoE/KDA, которых в этом рецепте нет. «Fused epilogue уже есть»
из задания — неверно: у dense-блока его нет, существующий отказывает при
`use_mhc=false`.**

---

## 5. Ранжирование кандидатов

### C1 — pointwise-шов блока: часть 36–40 мс не-GEMM остатка. Снимается 8–14 мс (5–10%)

Что это (измерено по оп-каталогу): на каждый блок fwd+bwd приходятся
(а) каст bf16→fp32 выхода каждой проекции, (б) `x + delta` в fp32, (в) LN,
(г) обратные касты. Верхняя оценка трафика — **52.5 ГБ/шаг, из них 29.1 ГБ
`_to_copy` (вес — только 0.57 ГБ) и 17.0 ГБ `add.Tensor`**. Рычаг — один
Triton-эпилог на блок (`cast + residual-add + LayerNorm`, fwd и VJP) и/или
bf16-остаток; в пределе это снимает 3–4 лишних прохода остатка на блок.

- **Ожидание:** 8–14 мс (5–10%). Не 36–40: attention и GELU/LN остаются, и часть
  кастов — цена смешанной точности, а не лишний проход.
- **Чем доказать:** сначала расширить census на transformer-форму (сейчас
  `scripts/pass_census.py` умеет только `LoopBlock`), затем ядро + parity-гейт по
  образцу `fused_residual.PARITY_RTOL/GRTOL` (1e-5 / 5e-5), затем 40-шаговая
  рука. Разделить шов и attention можно только профилем — две строки в
  `kernel_profile.py:188` (`TransformerBlock._block → "block"`,
  `F.scaled_dot_product_attention → "attention"`).
- **Файл:** `model/transformer.py:234-280` (+ новый файл рядом с
  `model/fused_residual.py`).
- **Риск:** fp32-остаток — это контракт численности порта; bf16-остаток требует
  A/B и **сейчас недоступен как флаг** (см. C7).

### C2 — не-NS оптимизатор: 29.7 мс (20%). Снимается 6–12 мс (4–8%)

Что это: momentum (bf16), `colrow_normalize_` (4 прохода), `mul_(flag)`,
writeback, AdamW. Трафик — **верхняя оценка ~12 ГБ** (тот же счёт по `numel`),
то есть ≥13.4 мс при 896 ГБ/с, а измерено 29.7 мс — **как минимум вдвое хуже
полосы**, то есть это мелкие ядра/занятость, а не байты.
Рычаг — один multi-tensor проход на группу формы вместо ~6 проходов.

- **Ожидание:** 6–12 мс (4–8%) при слиянии momentum+ColRow+flag+writeback.
- **Чем доказать:** регион `optimizer` в `kernel_profile.py` (seam уже есть,
  `kernel_profile.py:229`) **на этой рецепте — он никогда не снимался**; на
  m400b он был 23.5% шага. Затем ядро + гейт бит-в-бит с текущим
  `colrow_normalize_` (порядок важен: `tests/test_muon_fast.py`).
- **Файл:** `train/optim.py:1329-1590`.
- **Риск:** нумерация ColRow и bf16-момент пиннятся тестами; новый writer нельзя
  сверять с текущим fused-AdamW — он уже расходится (C3).

### C3 — сначала починить fused-AdamW writer (корректность, не скорость)

`_foreach_adamw_step` **включён по умолчанию** (`train/optim.py:2086`), и
`bench_compile`'s собственный `_opt_routes` печатает
**`bpb drift 2.47e-04` за 20 шагов при гейте 1e-4** и выходит с кодом 4
(`compile_f3.log`: «that is a bug, not a speed/accuracy trade»). У него же нет
ни одного счётчика — лог не может сказать, что он работал. Группа — 0.47M
параметров, так что «172.6 против 360.0 мс» к времени шага отношения почти не
имеет (и дельта не объясняется одной AdamW-группой — там же сидит несвязанная
стадия); чинить надо потому, что это другой оптимизатор на каждом прогоне.
**Файл:** `train/optim.py:2117-2185`. **Ожидание по скорости: 0.**

### C4 — `torch.compile`: измерено 0…+12%, не доказано; как включено — 0.49×

- `[M]` 40 шагов: 291.7 против 289.9 мс — **−0.6%** (`logs/thru/compile.jsonl`).
- `[M]` min-of-5, g=3: 457.2 против 470.0 — **+2.7%** (`compile_f3.log`).
- `[M]` `max-autotune-no-cudagraphs`: 413.1 — **+12.1%**, но `opt.step` в том же
  прогоне «съехал» 203.6→172.8 мс, а оптимизатор не компилировался: 30.8 из
  56.9 мс дельты — это дрожание несвязанной стадии.
- `[M]` **ветка, которую берёт сам тренер** — `torch.compile(model, dynamic=True)`
  (`train/loop.py:1474`) — измерена как `trainer` = **954.0 мс, 0.49×**. Это и
  есть «1.20× на другом шейпе» из комментария `config.py:1959-1960`.
- **Ожидание:** 0–8%. **Чем доказать:** `scripts/ab_compile.sh` (написан, **не
  запускался**) с `dynamic=False`/`max-autotune`, один процесс, min-of-5,
  600 шагов, плюс parity. **Файл:** `train/loop.py:1474`. **Риск:** поверить
  строке 413.1 и получить −51%.

### C5 — бэкенд attention: ожидание 0%, но это `[S]` до одной строки на карте

`transformer.py:267` вызывает `F.scaled_dot_product_attention` на bf16
`[8,13,1024,64]`, `is_causal=True`. `qk_norm` в блоке **нет**, `logit_softcap=0`
— «fused QK-norm/softcap path» фьюзить нечего: такого пути не существует.

- `[C]` math-путь стоит 20–50 мс только на attention (§3) и оставляет на весь
  остальной не-GEMM шов 16 мс в самом дешёвом углу — меньше, чем одни касты.
  Поэтому ожидание — fused (8–16 мс), с оговоркой `[S]`: арифметика сильно
  против math, но не запрещает его математически.
- `[M]` колесо собрано под `sm_120` (`torch.cuda.get_arch_list()`), flash
  доступен (`is_flash_attention_available() == True`), в `libtorch_cuda.so` есть
  `flash_attention_forward/backward`.
- **Чем закрыть:** на карте — `with torch.nn.attention.sdpa_kernel(
  [SDPBackend.FLASH_ATTENTION]):` один forward, или `torch.profiler` на 5 шагов
  искать `flash_fwd`/`cudnn_attention`. 5 минут.
- **Файл:** `model/transformer.py:267`. **Риск:** если fused уже стоит — ноль;
  если нет — 10–15% и смена численности backward (flash недетерминирован).

### C6 — `grad_accum=3` на быстрой руке: +35% пропускной, не времени шага

Сейчас лента C — g=1: 56.2 кБ/с, и **39% шага платится за один микробатч**.
При g=3 тот же оптимизатор амортизируется на три:

```
3 × 88.8 + 57.0 = 323.4 мс на 24 576 Б → 76.0 кБ/с   (+35% к 56.2)
```

- `[M]` амортизация уже видна на медленной руке: g=1 289.9 мс (28.3 кБ/с) против
  g=3 480 мс (**51.2 кБ/с, +81%**), `logs/night_opt/mix_s1.jsonl` / `logs/flagship3.jsonl`.
- **Чем доказать:** одна 40-шаговая рука в `throughput_ladder.sh` с
  `--set grad_accum=3`; затем 2k-шаговый BPB, потому что эффективный батч 24K
  требует другого LR. `scripts/ab_batch_lr.sh` написан и **не запускался**.
- **Риск:** не бесплатно по качеству — это другой режим оптимизации (LR).
  Время шага не меняется; меняется число шагов на байт. Сама цифра 76 кБ/с —
  `[C]` из измеренных g=1 и g=3; её перенос на быструю руку — `[S]`, пока рука
  не запущена: предполагается, что fwd+bwd на микробатч не меняется при g=3.
  Отдельно fwd+bwd при g=3 на этой форме **не измерен**; на медленной руке
  амортизация видна (+81% пропускной), но её величина включает разницу
  min-of-reps (`bench_compile`) против среднего окна лога.

### C7 — `residual_dtype` на этой форме — молчаливый no-op (дефект, не рычаг)

`residual_dtype="bf16"` **принимается** с `block_form="transformer"` и **не
читается никем**: поле живёт в `LoopBlock` (`loopblock.py:288-295`), у
`TransformerBlock` нет даже атрибута (проверено: `hasattr(block,
"residual_dtype") == False`). То есть 5–8% из «bf16-остатка» **недоступны
флагом**, а флаг врёт. Либо провести его в блок (это C1), либо отказать на этой
форме в `validate_block_form` (`config.py:3078`).

---

## 6. Чего НЕ делать (каждое с артефактом)

| не делать | почему | артефакт |
|---|---|---|
| CUDA-графы (`opt_graph`/`full_graph`) | три независимых замера: +3%, −4%, 0; на этой ленте 0.99× и 0.99× | `docs/records/reviews/2026-11-05-graph-verdict.md:9-21`; `the-16x.md:36-38`; `thru/graph.jsonl` |
| «2.3% на final_byte» как аргумент за граф | это запись в памяти, **лога нет**; удалённый аудит цитирует её как «measured» | `docs/records/reviews/2026-11-08-deletion-audit.md:137` |
| `muon_state_dtype=bf16` ради скорости | 292.4 против 289.9 мс = **−0.9%**; это memory-рычаг | `thru/state_bf16.jsonl` |
| `ns_steps=3` | 2.08×, но **+0.261…+0.274 BPB**, вне шума сидов (5.4σ) | `the-16x.md:34-35`; `docs/records/reviews/2026-11-08-data-eval-integrity.md:465-472` |
| …и `ns_steps` — PROGRESS KEY | resume может молча переключить 5→3 и потерять 0.27 BPB; отказа нет | `config.py:85` |
| `ns_kind=triton` | `MAX_RANK=64`; живые группы `min(m,n)=832` → **COUNTED fallback на всех четырёх** | `train/ns_kernels.py:140, 216-244` |
| `mm_dtype=fp8` | у `_scaled_mm` нет backward; тренировочный forward падает в bf16 COUNTED, скорость не меняется | `train/loop.py:1160-1165` |
| `optim_8bit` | громкий отказ: нет пути для device-флага | `train/loop.py:1338-1342` |
| батч 16/32 | +11% / +6.5%; b32×s1024 не влезает в 16 ГБ; M=8192 уже 0.91 волны на 70 SM | `AGENTS.md` §3.1; `docs/records/reviews/2026-11-08-mfu-first-look.md` §1.6 |
| загрузчик данных | уже `frombuffer`: 0.08 мс/шаг после 0.74; на 146 мс это 0.05% | `train/data.py:89`; `docs/records/reviews/2026-11-08-cpu-perf-audit.md:294-315` |
| число запусков как рычаг | 1 700/шаг, хост 8–13 мс, 0 синков; «GPU residue ≈0» был на 9.2M KDA, не здесь | `docs/records/reviews/2026-11-08-cpu-perf-audit.md:64-72, 146-148` |
| `full_graph` | отказывает с `doc_masking=true`, а живая лента его использует | `config.py:4752` |
| `F.cross_entropy` вместо `log_softmax+gather` | экономит 8.4 МБ (0.01 мс) и не бит-идентичен | `docs/records/reviews/2026-11-08-cpu-perf-audit.md:392-394` |
| цена pointwise-прохода «по 1.9 ТБ/с» | это вдвое завышенная полоса: у 5070 Ti **896 ГБ/с** GDDR7, не HBM (28 Гбит/с × 256 бит; `nvidia-smi` 14 001 МГц) | `scripts/pass_census.py:68` |

---

## 7. Чего этот аудит не измерил (и чем закрыть)

1. **Профиля ядер на этой форме не существует.** Единственный записанный
   `kernel_profile` — m400b (`/tmp/opencode/kprof.json`, 199M, MoE/mHC/KDA):
   `optimizer` 23.5%, `moe_dispatch` 21%, 4 692 запуска. У него **нет** региона
   для `TransformerBlock`, `SDPA` и `mm_mixed` (`kernel_profile.py:110-124,
   188-232`), поэтому на плотной форме весь блок уйдёт в `other`. Две строки в
   `SEAM_SPECS` превращают вопрос §3 из арифметики в профиль.
   Команда (когда карта свободна):
   `uv run --extra gpu python scripts/kernel_profile.py
   --checkpoint /tmp/opencode/flagship3/flagship3.pt --json /tmp/kprof_f3.json`
   (снимок шага 5000 на диске есть).
2. **Разделение «шов vs attention»** — единственное, чего не даёт арифметика
   §3: 36–40 мс не-GEMM остатка содержат и то и другое. Профиль из п.1 решает
   это одним прогоном.
3. **Регион `optimizer` на этой рецепте не снимался.** 29.7 мс не-NS блока —
   это измеренная *дельта*, а не измеренная причина; разложение на momentum /
   ColRow / writeback / AdamW появится только из п.1.
4. **Точное `F`** (модель без clip/firewall) — 88.8 мс из `bench_compile`, где
   firewall и clip отсутствуют. В тренере их +3–5 мс; на общий вывод это не
   влияет, но «88.8» — это минимум, а не среднее лога.
5. **Время шага по логу не пишется** без `--timers`; живые логи дают только
   оконный `tokens_per_s` в JSONL. Это не дефект, но `ms/step` в отчёте — всегда
   производная от `bytes_per_step / tokens_per_s`, а не поле.
6. **Константа полосы в `pass_census.py:68`** (1.9 ТБ/с) неверна для этой карты
   вдвое; её надо исправить до следующего census, иначе каждый census-вывод в
   `docs/` (включая `fused_kda.py:1-40` и `pass_census`-раздел
   `2026-11-05-kernel-profile.md`) занижает цену pointwise-проходов в 2.1×.
   Перекрёстная проверка: единственный записанный профиль ядер (m400b,
   `/tmp/opencode/kprof.json`) даёт 182 ГБ за 200 мс GPU-busy = **910 ГБ/с** —
   потолок 896, а не 1.9 ТБ/с.

---

## 8. Воспроизведение (всё CPU, < 5 мин)

```bash
cd /home/sehaxe/dormouse/port
# 1. что реально резолвит файл рецепта
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python - <<'PY'
from dormouse.config import load_run
r = load_run("configs/flagship3_dense.toml")
print(r.train.seq_len, r.train.grad_accum, r.train.ns_steps, r.train.ns_dtype,
      r.train.muon_state_dtype, r.train.muon_lr_mult, r.train.ckpt_every)
PY
# 2. census шага (meta): 2 639 aten-опов / 1 384 запуска / 5 663.7 GFLOP / 52.5 ГБ шва
#    (скрипт этого аудита: /tmp/opencode/f3_census.py — в репозиторий не кладётся)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python /tmp/opencode/f3_census.py
# 3. измеренные руки
python3 - <<'PY'
import json, glob
for p in sorted(glob.glob("/home/sehaxe/logs/thru/*.jsonl")):
    r = [json.loads(l) for l in open(p) if '"log"' in l][-1]
    print(f"{p.split('/')[-1][:-6]:18s} {r['tokens_per_s']:8.0f} B/s  {8192/r['tokens_per_s']*1e3:6.1f} ms")
PY
# 4. бюджет: 145.8 мс, 51.5 TFLOP/s, не-GEMM остаток 36.5–39.8 мс
python3 -c "ms=8192/56181.47483904764*1e3; print(ms, 7516/ms, 88.8-4.659/95*1e3, 88.8-4.659/89*1e3)"
```

---

## 9. Итог одной строкой (RU)

**Дальше нет одной кнопки на 10–20%. Есть два честных пути, оба — новое ядро:
снять ~8–14 мс трафика с pointwise-шва блока (Triton-эпилог + bf16-остаток,
5–10%) и ~6–12 мс с мелких ядер оптимизатора (один fused-проход, 4–8%). Вместе
это 10–18% — ровно тот диапазон, который искали, но не одним флагом. Attention —
ожидание 0% (fused), подтверждается одной строкой на карте; графы/compile/state
bf16 измерены как ноль, `ns_steps=3` стоит +0.27 BPB, triton-NS отказывает по
`MAX_RANK=64`.
Единственный выигрыш ≥10%, доступный сегодня, — `grad_accum=3`: +35% пропускной
способности, не времени шага.**
