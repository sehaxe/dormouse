# Compression as the compute unit: NCM re-read, the 2023–2026 landscape, the information bound, and the codec inequality

**Date:** 2026-11-07. **Mode:** read-only — arXiv full texts and tables, the released NCM code/config, repo docs and logs; no GPU run, no code edit.
**Marks:** **[T]** number read from a paper table · **[A]** abstract/text only · **[C]** computed here from stated numbers (formula shown) · **[M]** measured in this repo (path or log named) · **[S]** SPECULATION, no direct measurement.

**Scope.** The owner's metric is **BPB per (GiB × hour)** at 95M on one 16 GiB card (`.bulba/goal.md`), not BPB per byte at equal parameters. The brief asks four things: re-read NCM, survey every “compression as the compute unit” line 2023–2026, do the information arithmetic, and solve the codec inequality. This report answers all four and audits the three numbers in the brief that do not match the artifacts.

---

## 0. TL;DR (RU)

1. **NCM — по-прежнему v1, 2026-10-09; пересмотра нет.** Кодек 111.2M против генератора 94.65M на OWT (111.4M vs 97.07M на LM1B), 0 encoder-блоков, 1–4 decoder-блока, K=12–32, V=4096–8192. Реконструкция 98.79–99.99%, не lossless. 21× — против DUO (диффузия, 11× меньше NFE), 31× — против AURORA-LM; против AR нет ни одного wall-clock, а единственное AR-сравнение (WMT14, token-level) NCM проигрывает 22.6 vs 25.2 при равном NFE и сравнивается только с MBR при 2× NFE (132 vs 64). **Против AR byte transformer при равных параметрах/FLOPs — не измерено.**
2. **Новое, чего не было в прошлом отчёте.** Релизный конфиг даёт точное расписание уровней. OWT: 1,4,9,…,1024 (K=32) → **11 440 кодов на 1024 токена = 11.2 позиции/токен = 134 бит/токен = 33.5 бит/байт** (V=4096). LM1B: 1,8,…,128 (K=12) → **641 код на 128 токенов = 5.0 позиций/токен = 65 бит/токен = 14.5 бит/байт** (V=8192). Текст несёт 1.46–2.53 бит/байт (наш замер, §3). **Код NCM — избыточное представление в 5–25× энтропии текста, а не сжатие.** Поэтому 21× — выигрыш латентности (32 шага вместо 128), не FLOPs.
3. **Ландшафт 2023–2026: положительные equal-FLOP результаты при ≤1B есть, но маленькие.** H-Net 760M −0.013 BPB при +2.4% FLOPs; ByteFlow 600M −0.02…−0.03 BPB при +5% FLOPs; Scratchpad 2B 2.9× FLOPs при паритете с byte-базой; TST 270M −0.070 loss при равных шагах. **При 95M — ни одного.** Крупные выигрыши (BLT +1.1 на 8B, MegaByte +0.057 при 3.2× параметров, NCM 21×, ByteFlow +3.04 на 1.3B) — на ≥1.3B или против диффузии/byte-базы.
4. **Информационная арифметика (наш корпус, 256 МиБ).** H₂=9.03, H₄=14.47, H₈=20.22, H₁₆=23.37 бит/блок; 4.51/3.62/2.53/1.46 бит/байт для k=2/4/8/16; xz -9 = 1.94 бит/байт. Кодбук V=4096 покрывает 99.40% 2-байтовых блоков, 58.85% 4-байтовых, **16.07% 8-байтовых**, 2.54% 16-байтовых. Для 99.9% покрытия при k=8 нужно 9.36M кодов; для 100% — 9.39M (на выборке), т.е. 2^H₈=1.23M — необходимое, но не достаточное условие. **Ни одна опубликованная latent-text схема при 95M не удовлетворяет lossless-границе для k=8 с малым V.** NCM обходит её, тратя 134–145 бит/токен (избыточность), а не сжимая.
5. **Неравенство кодека.** При k=8 и равных байтах на шаг 3× невозможно при измеренной f=0.64 (потолок 1.46×); для 3× нужно f≤0.238 даже с бесплатным кодеком. В режиме равных позиций (8× байтов) кодек может стоить до **20.8% базовой цены байта = 144 MFLOP/байт**, или 167% шага основной модели. Наш conv-кодек — 2.14 MFLOP/байт (2.9% FLOPs, 5.4% wall-clock), causal-conv с маскированным входом — ~0.25 MFLOP/байт. **Кодек не является ограничением; ограничение — качество на байт.**
6. **Три эксперимента.** E1 — triage «слепота vs потеря информации» (6–8 GPU-h); E2 — equal-GPU-hours контроль (27.8 h); E3 — byte-bag/TST (5.6 h). E1 первый, потому что он решает, имеет ли смысл E2.

---

## 0.1 Аудит: три числа из брифа не совпадают с артефактами

| Утверждение брифа / `.bulba/goal.md` | Что в артефакте | Вердикт |
|---|---|---|
| «2.04× скорость шага» | Ни один лог/док не печатает 2.04. Ближайшее: 47.6 → 32.6 ms = **×1.46** [M, `docs/reviews/2026-11-03-three-verdicts-and-a-leak.md` §3]; bf4→bf8 при 150M: 23.1k → 24.3k поз/с = **×1.05** [M, `~/logs/ab_bf400/*.jsonl`] | **Не подтверждено.** 2.04 в артефактах отсутствует |
| «−0.3725 BPB на 199M» | `~/logs/night2/ab_byteflow.log:133`: control 3.6152 ± 0.0202, channel 3.2427 ± 0.0099, Δ −0.3725. Шапка: **1.9M params, b4 s512, 600 шагов, eval 40 960 B**; channel медленнее: 70.1 vs 53.6 ms/step | **Число верное, масштаб неверный.** Это 1.9M, не 199M; и это attention-кодек, не conv |
| «conv codec lost 1.5 BPB at 199M» | `~/logs/ab_bf400/bf8_s1.log`: 4.135 BPB, **150.8M** params, 800 шагов; `full_s1.log`: 2.684 BPB, **199.3M** params. Разница 1.45 BPB, но у conv-плеча нет Engram/mHC/CSA/LayerRoPE и на 48M меньше параметров | **Число верное, причинность не установлена.** Это конфаундированная full-recipe дельта, а не эффект рецептивного поля |
| «conv codec 357k params, 5.4% шага» | encode 0.84 + decode 0.91 ms из 32.6 ms = **5.37%** [M, `2026-11-03` §3]; 357 120 против 93 909 248 параметров, 1.27–1.90% MAC | **Подтверждено** (но шаг launch-bound: 2.9% по FLOPs, 5.4% по wall-clock) |

**Отсюда правило для всего отчёта:** «−1.5 BPB» — не измерение слепоты кодека. Чистого A/B «causal conv vs byte» при 95M–199M в репозитории **нет**. Это эксперимент E1.

---

## 1. NCM, перечитанный (2610.14162v1, 2026-10-09)

### 1.1 Объект

| Поле | Значение | Источник |
|---|---|---|
| arXiv id | **2610.14162** | abs page |
| Версия | **v1** (единственная; v2 нет на 2026-11-07) | abs page, submission history |
| Дата | **2026-10-09 17:49:45 UTC** | abs page |
| Категория | cs.CL (cs.AI, cs.LG) | abs page |
| Авторы | Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré | abs page |
| Код | `github.com/HazyResearch/ncm`, MIT, создан 2026-10-11, push 2026-10-14, 1493 stars, 190 forks, 726 KB | GitHub API [C] |
| Блог | `hazyresearch.stanford.edu/blog/2026-10-12-ncm` — повторяет цифры статьи, BPB/AR-сравнения не добавляет | [A] |

**Вывод по версии:** прошлый отчёт читал ту же v1. Пересмотра, который менял бы числа, нет. Новое — только релиз кода и конфигов после отчёта.

### 1.2 Кодек: архитектура и цена

| Параметр | OWT | LM1B | TinyStories | XSum | WMT14 | Sudoku | Источник |
|---|---:|---:|---:|---:|---:|---:|---|
| Уровней K | 32 | 12 | 12 | 8 | 11 | 12 | [T, Table 5] |
| Codebook V_k | 4096 | 8192 | 4096 | 4096 | 4096 | 16 | [T, Table 5] |
| Latent width d | 384 | 384 | 384 | 384 | 384 | 256 | [T, Table 5] |
| Encoder blocks | 0 | 0 | 0 | 0 | 0 | 0 | [T, Table 5] |
| Decoder blocks | 1 | 4 | 1 | 1 | 1 | 1 | [T, Table 5] |
| Tokenizer params | **111.2M** | **111.40M** | 36.43M | 60.63M | 144.00M | 14.90M | [T, Table 5] |
| Generator params | **94.65M** | **97.07M** | 88.61M | 88.02M | 119.96M | 25.46M | [T, Table 6] |
| Tokenizer ÷ generator | **1.17×** | **1.15×** | 0.41× | 0.69× | 1.20× | 0.59× | [C] |

**Что это значит.** «Encoder blocks = 0» — это не опечатка: в релизном коде `enc_layers=0 / dec_layers=1 reproduce the original architecture: encoder = wte+wpe lookup, decoder = one Block` [C, `models/vqvae.py`]. То есть «кодек» — это embedding + downsampling + один (или четыре) decoder-блока; 111M параметров — это в основном codebook-таблицы (32 × 4096 × 384 = 50.3M) и embedding/выходной слой (50 304 × 384 = 19.3M дважды). **Кодек не тратит 111M параметров на вычисление; он тратит их на таблицы.**

**Расписание уровней — то, чего нет в Table 5.** Релизные конфиги:

| Датасет | `manual_levels` | K | Σℓ_k | Кодов/токен | Бит/токен (V) | Бит/байт (B/токен) | Источник |
|---|---|---:|---:|---:|---:|---:|---|
| OWT | 1,4,9,16,…,1024 (k²) | 32 | **11 440** | **11.17** | **134.1** (V=4096) / 145.2 (V=8192) | **33.5 / 36.3** (4.0) | [C, `config/eval/owt.yaml`] |
| LM1B | 1,8,16,24,32,40,52,64,76,92,108,128 | 12 | **641** | **5.01** | **60.1** (V=4096) / 65.1 (V=8192) | **13.4 / 14.5** (4.5) | [C, `config/eval/lm1b-small.yaml`] |

**Ключевое следствие.** ℓ_K = L (последний уровень имеет один код на токен). Поэтому Σℓ_k ≥ L + K(K−1)/2 > L: **многоуровневый residual VQ не может иметь меньше позиций, чем токенов.** OWT: 11.2 позиции на токен; LM1B: 5.0. В пересчёте на байт OWT — 2.79 позиции/байт, LM1B — 1.11. Против byte-AR (1 позиция/байт) это **2.79× / 1.11× FLOPs** при той же ширине; против token-AR (0.25 позиции/байт при 4 B/токен) — **11.2× / 5.0×**. Это не сжатие вычислений; это перераспределение их в параллельные шаги.

### 1.3 Пол реконструкции

| Датасет | Exact token accuracy | Ошибок на токен | Escape-cost floor, BPB (12.5 бит/замена, 4–4.5 B/токен) | Источник |
|---|---:|---:|---:|---|
| OWT | **99.99%** | 1e-4 | 0.00031 | [T, Table 1] + [C] |
| LM1B | **99.86%** | 1.4e-3 | **0.00389** | [T, Table 1] + [C] |
| TinyStories | 99.98% | 2e-4 | 0.00062 | [T, Table 1] + [C] |
| XSum | 99.79% | 2.1e-3 | 0.00656 | [T, Table 1] + [C] |
| WMT14 | **98.79%** | 1.21e-2 | **0.03781** | [T, Table 1] + [C] |
| Sudoku | 100.00% | 0 | 0 | [T, Table 1] |

**Что это значит и чего это не значит.** 12.5 бит/замена — это **[S]**, не измерение статьи: статья сообщает exact-match, а не NLL декодера. Реальная цена кодека — это `log p_ψ(x|z)` (Eq. 17), и она **не опубликована**. Exact-match — это 0/1-лосс, а не bits per token; поэтому 0.0003–0.038 BPB — это **нижняя граница стоимости escape-канала**, а не полная стоимость кодека. Прошлый отчёт называл 0.003–0.005 BPB «полом»; это верно только для LM1B и только при допущении 10–15 бит/замена. Для WMT14 пол в 10× выше.

**Второй, больший пол — избыточность кода.** 134 бит/токен на OWT при энтропии токена ~6–8 бит (1.5–2 бит/байт × 4 B) — это 17–22× энтропии. Кодек не «сжимает и теряет»; он **расширяет** представление и потом восстанавливает из него текст. Именно поэтому exact-match высокий: у кодека много битов, а не потому, что кодбук эффективен.

### 1.4 Скорость: что именно побито

| Сравнение | NFE NCM | NFE baseline | Ускорение | Что за baseline | Источник |
|---|---:|---:|---:|---|---|
| LM1B vs DUO | 12 | 128 (11× меньше) | **21×** | дискретная диффузия, полная последовательность на каждом шаге | [T, §4.6; Table 7] |
| OWT vs сильнейший DLM | — | — | **10×** | DLM | [A, §4.6] |
| XSum vs AURORA-LM | 8 | 128 (16× меньше) | **31×** | flow/diffusion, полная последовательность | [T, §4.6] |
| WMT14 vs AR (token, 99M) | 11 | 64 | **0.94× по NFE, но хуже качество**: 22.6 vs 25.2 BLEU | AR token-level, не byte | [T, Table 2] |
| WMT14 vs AR + MBR | 132 | 64 | **2.06× NFE за равный BLEU 25.2** | AR token-level | [T, Table 2] |

**Чего в таблице нет.** Ни одного wall-clock против AR-модели; ни одного сравнения с AR **byte** transformer; ни одного BPB. 21×/10×/31× — это против DLM, которые делают 64–128 полных проходов по последовательности. Против AR token-level на WMT14 NCM **проигрывает при равном NFE** и сравнивается только с MBR при 2× NFE. Это и есть ответ на «10–31× быстрее AR»: **такого измерения не существует.**

### 1.5 Качество против AR byte transformer при равных параметрах и FLOPs

**Ответ: не измерено.** В статье нет AR byte baseline; единственный AR — token-level 99M на WMT14 (Table 2). Нет BPB, нет bits-per-code, нет FLOPs-таблицы. Метрика — Gen. PPL под GPT-2-large, а не BPB. Следовательно:

- «равные параметры» — только по генератору (94.65M vs 139M DLM); системно 94.65 + 111.2 = **205.9M**, т.е. 1.48× baseline [C].
- «равные FLOPs» — не заявлено и не измерено; при K=12 NCM ~1.11× byte-AR по позициям, при K=32 ~2.79× [C].
- «качество не потеряно» — Gen. PPL не сравнимо с нашим BPB; на WMT14 при равном NFE NCM теряет 2.6 BLEU.

### 1.6 Вердикт: прошлый отказ всё ещё верен?

**Да, для 95M/1 GiB — но по трём причинам, а не по одной.**

1. **Системная цена.** Кодек 111.2M/111.4M больше генератора 94.65M/97.07M; системно 205.9M/208.5M. «Matched parameter scales» верно только для генератора. [T, Tables 5–6]
2. **Код избыточен.** 134–145 бит/токен (OWT), 60–65 бит/токен (LM1B) против энтропии текста 1.46–2.53 бит/байт. Это 5–25× энтропии; это не компрессия, а overcomplete-представление. [C, §3]
3. **Нет AR/byte-сравнения.** Единственное AR-сравнение — token-level WMT14, где NCM проигрывает при равном NFE (22.6 vs 25.2) и требует 2× NFE + MBR для паритета. [T, Table 2]

**Две поправки к прошлому отчёту.**
- Кодек — **не per-step cost**: коды предвычисляются токенизатором, генератор учится на готовых кодах. Поэтому 111M — это стоимость системы и обучения токенизатора (10k–120k обновлений, batch 768–4096), а не стоимость шага. Прошлый отчёт правильно назвал это «стоимостью frozen pretraining run», но таблица «36–144M против 88–120M» читается как per-step; это не так.
- «0.003–0.005 BPB floor» — это **LM1B-оценка escape-стоимости**, не универсальный пол. Для OWT 0.0003, для WMT14 0.038; и decoder NLL не измерен вообще.

### 1.7 Что должно было бы быть правдой для победы при 95M / 1 GiB

| # | Условие | Почему сейчас не выполнено | Что нужно |
|---|---|---|---|
| 1 | Токенизатор ≤10M параметров и обучен на ≤1 GiB | 111.2M, обучен на 13.36B (LM1B) – 127.22B (OWT) позиций | Сжать кодек в 11×; переобучить на 1 GiB без переобучения |
| 2 | Σℓ_k ≤ L_bytes (не больше позиций, чем у byte-модели) | ℓ_K = L_tokens; Σℓ_k ≥ L_tokens + K(K−1)/2; OWT 2.79 позиции/байт | Отказаться от ℓ_K = L (потеря точности) или от байтового входа |
| 3 | Генератор бьёт byte-AR при равных FLOPs | Не измерено; при K=12 FLOPs ~1.11× byte-AR — единственный честный тест | Прямой A/B на BPB при равных FLOPs |
| 4 | 100% byte reconstruction | 98.79–99.99% token-level; для кода/разметки/цифр этого мало | Lossless-кодек или escape-канал с измеренным NLL |
| 5 | Кодбук удовлетворяет энтропийной границе без 5–25× избыточности | Нет: код тратит 134–145 бит/токен | Фиксированный k-байтовый кодбук с V ≥ 9.4M (k=8) — или признать, что это не компрессия |

**Итог:** NCM — это метод **латентности и параллелизма**, а не метод сокращения вычислений на байт. Для нашей цели (BPB на GiB×час при 95M/1 GiB) он не даёт ни одного измеренного преимущества; отказ остаётся.

---

## 2. Ландшафт 2023–2026: каждая линия «сжатие как единица вычисления»

`Codec cost` — доля шага (FLOPs, или wall-clock, где статья его печатает), которую платят байт→патч/локальные модули. `Match` — что уравнено: **iso-FLOP**, **iso-param+tokens**, **iso-quality**, **fixed data**. Где число выведено из гиперпараметров статьи — [C]. Где статья не печатает FLOPs — «not reported».

| # | Кандидат (id, версия, дата) | Единица | Цена кодека, доля шага | Match | Измеренная дельта при ≤1B | Кред. | Усилие |
|---|---|---|---|---|---|---|---|
| 1 | **NCM** — `2610.14162v1` (2026-10-09) | 32-уровневый RVQ, 1–1024 кода, V=4096 | Токенизатор 111.2M (1.17× генератора); 0 enc + 1–4 dec блоков; **не в шаге** (коды предвычислены) | Только по параметрам генератора; системно 1.48×; **не iso-FLOP** | Нет AR/byte-сравнения; 21× vs DUO (DLM); reconstruction 98.79–99.99% | Stanford/Together, код релизнут; независимой репликации нет | Высокое: обучение токенизатора 13–127B позиций |
| 2 | **BLT** — `2412.09871v1` (2024-12-13) | entropy patches 4.5/6.1/8 B | Enc 12M + dec 113M при 1B; **36% FLOPs при p=4.5, ~50% при p=8** [C: 125/(125+1000/p)] | **iso-FLOP** | ps=8 **хуже** BPE при 1B; лучше при 7B; 8B avg +1.1 vs Llama3 | Meta, таблицы + scaling; независимой репликации нет | Высокое: local/global стек, 150B байт |
| 3 | **MegaByte** — `2305.07185v2` (2023-05-19) | fixed 8 B | Local 262M + global 758M; **73.4% FLOPs** [C: 262/(262+758/8)]; всего **1.11× FLOPs/byte** | **compute+data matched, params NOT** (1.02B vs 320M) | PG-19 BPB **1.057→1.000**; code 0.575→0.411; arXiv 0.816→0.678 | Meta, Table 2 | Высокое: two-level model |
| 4 | **SpaceByte** — `2404.14408v3` (2024-10-06) | space boundaries, 6–8 B | Local blocks = global blocks; ~50% слоёв [C] | **fixed training+inference compute** | «roughly matches tokenized»; BLT: «remains far from Llama 3» | Academic; BLT-репликация противоречит | Среднее |
| 5 | **H-Net** — `2507.07955v2` (2025-07-15) | learned 4.8–5.3 B | **0.42→0.43 GFLOP/byte (+2.4%)** | **iso-FLOP + data** | 760M: **0.756→0.743**; 1.3B: 0.730→0.715 | Academic, чистая таблица | Высокое: hierarchical chunker |
| 6 | **Scratchpad** — `2605.09630v1` (2026-05-10) | 4–16 B + intra-patch scratchpad | SP стоит **1.97×** vanilla patched [C: 5.7/2.9]; net **2.9×** vs byte при p=16 | **fixed data, measured FLOPs/byte** | byte 54.1; p=16 48.0; **p=16+SP 54.2**; p=8+SP 53.8 | Google, ~2B, 400B байт | Средне-высокое |
| 7 | **TST** — `2605.06546v2` (2026-05-19) | bag of 6–16 tokens | **0** (embedding average + multi-hot CE) | **equal steps/FLOPs** | 270M **3.212→3.142**; 600M 3.019→2.943; 3B **2.808→2.676**; 10B **2.5× time** | Industry, 270M–10B | Низкое: drop-in loss |
| 8 | **Byte-LM scaling** — `2610.05978v1` (2026-10-05) | 1 B/pos + TST(4) + hash n-grams | **0** | **iso-FLOP и iso-param** | При равных FLOPs **subword всегда лучше**; при равных optimal params byte лучше; byte 3.2× байт/параметр | Academic, 200M–3B | Низкое: TST + hash |
| 9 | **ConceptLM** — `2602.08984v1` (2026-02-09) | VQ над hidden states; **BPE на входе** | Concept head; −11.3% train time на 8B | **params matched** | GPT2-124M PPL **101.94→84.75**; Pythia-70M +3.6; 410M +0.8 | Academic, 70M–1.5B | Среднее |
| 10 | **LCM** — `2412.08821v2` (2024-12-15) | SONAR sentence embedding | Sentence encoder/decoder; FLOP-выигрыша нет | **не FLOP-matched** | 1.4B smaLlama ROUGE-L **34.88 vs 33.64** у LCM; coherence 0.984 vs 0.938 | Meta, 1.6B/7B | Высокое; это не compute-unit |
| 11 | **DLCM** — `2512.24617v2` (2026-01-05) | variable-length concepts, R=4 | Compression снижает attention; +2.69% при matched **inference** FLOPs | **inference FLOPs matched; params 2.3B vs 1.3B** | **43.92 vs 41.23** (+2.69) на 12 zero-shot, 1T tokens | Academic, 274M–2.3B | Среднее |
| 12 | **ByteFlow** — `2603.03583v1` (2026-03-03) | learned 2.0–5.12× chunks | 6-layer local enc + symmetric dec; **1.07e21 vs 1.02e21 FLOPs (+5%)** | **FLOPs matched at 0.6B** | 600M: **BPB 0.86 vs 0.89**; +1.74 avg; 1.3B **+3.04** | Academic/industry, 600M/1.3B | Средне-высокое |
| 13 | **Proxy Compression** — `2602.04289v2` (2026-02-04) | external compressor views (gzip/tokenizer/neural) | Только обучение; на инференсе отбрасывается | Code domain; compute-matched | Выигрыш растёт с масштабом; **gzip fails**; 0.5B–14B | Academic | Среднее |
| 14 | **Lossy Horizon** — `2510.22207v1` (2025-10-25) | lossy text codec (EPC/VQ+RE) | MLM-декомпрессор; 1.1–2.8 BPC при 98% char fidelity | **rate–distortion, не pretraining** | EPC доминирует VQ+RE; 2% char error стоит 1.1–1.3 BPC | Academic | Низкое; не pretraining-unit |

### 2.1 Что таблица говорит (RU)

1. **Положительные equal-FLOP результаты при ≤1B есть, но они маленькие.** H-Net 760M: −0.013 BPB при +2.4% FLOPs. ByteFlow 600M: −0.02…−0.03 BPB при +5% FLOPs. Scratchpad 2B: 2.9× FLOPs при паритете с byte-базой. TST 270M: −0.070 loss при равных шагах. **Ни одного при 95M.**
2. **Крупные выигрыши — на ≥1.3B или против слабой базы.** BLT +1.1 avg на 8B (при 1B ps=8 хуже); MegaByte +0.057 BPB, но 3.2× параметров; NCM 21× против DLM; ByteFlow +3.04 на 1.3B и 500B байт (4× Chinchilla).
3. **Цена кодека не коррелирует со знаком результата.** TST/Byte-LM — 0% кодека, положительны/отрицательны по-разному. H-Net/ByteFlow — 2–5%, положительны. MegaByte — 73%, положителен только по параметрам. Наш 5.4% — в одном диапазоне с ByteFlow; разница в качестве, не в цене.
4. **Дискретная линия (NCM, VQ) — самая дорогая и наименее FLOP-matched.** Единственная text-VQ работа — NCM; остальное — аудио, KV-cache, embedding compression. **[Отрицательный результат поиска]**
5. **Sentence-level линия — не byte/patch codec.** LCM проигрывает 1.4B token-модели на ROUGE-L; DLCM выигрывает +2.69, но с 1.77× параметров (2.3B vs 1.3B), т.е. это не efficiency-win.

### 2.2 Отрицательные результаты (чего нет)

- **Нет RVQ/multi-level VQ как единицы pretraining текста вне NCM.** Поиск `all:"residual vector quantization" AND all:"language model"` даёт KV-cache (2410.15704), embedding compression (2510.12721), аудиокодеки; `all:"text codec"` — lossy compression (2510.22207); `all:"token compression"` — почти全部 multimodal. **[Отрицательный результат, не доказательство]**
- **Нет BPB и bits-per-code для генератора NCM** ни на одном масштабе. [T, все таблицы]
- **Нет BLT ps=8 ablation при 95M–470M.** Table 10 — 400M–8B; ps=8 vs ps=6 — 400M–3.6B с global, выросшим в 1.6–1.7×. [T, §5.3]
- **Нет text codec с lossless 8-байтовым фиксированным кодбуком при V≤65k.** Наш замер: V=4096 покрывает 16.07% 8-байтовых блоков. [C, §3]
- **Нет 95M positive для любого кодека.** Самый низкий положительный — TST 270M (token-level, без кодека) и ByteFlow 600M (2–5 B chunks, +5% FLOPs).

---

## 3. Информационный аргумент

### 3.1 Определения, корпус, оценщик — чтобы воспроизвести

| Параметр | Значение |
|---|---|
| Корпус | `/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_filtered/corpus.bin`, 19 881 989 922 B |
| Выборка | 8 × 32 MiB = **268 435 456 B**, смещения `[1225847154, 3711095894, 6196344634, 8681593374, 11166842115, 13652090855, 16137339595, 18622588335]` |
| Оценщик H_k | неперекрывающиеся k-байтовые блоки; plug-in `H = −Σ p log₂ p`; поправка Миллера–Мадоу `H_MM = H + (K−1)/(2N ln2)` |
| Смещение | plug-in **занижает** энтропию на конечной выборке; поэтому H_k — консервативная (нижняя) оценка требования. Для k=16 выборка недоопределена (14.0M различных из 16.8M блоков; 12.96M синглтонов), H₁₆ — нижняя граница |
| Верхняя граница скорости | `xz -9` на 32 MiB = **1.940 бит/байт**; это upper bound на entropy rate (компрессор не может быть ниже энтропии) |
| Распределение байт | H₁ = **5.063 бит/байт**; top-1 байт = 17.52% (пробел) |

### 3.2 H_k: энтропия k-байтового блока

| k | Блоков в 256 MiB | Различных | H_k plug-in (бит) | H_MM (бит) | H_k/k (бит/байт) | 2^H_k | V для 99.9% | V для 100% выборки | Покрытие V=4096 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 268 435 456 | 205 | **5.063** | 5.063 | 5.063 | 33.4 | 122 | 205 | 100% |
| 2 | 134 217 728 | 18 302 | **9.027** | 9.027 | **4.513** | 523 | 8 136 | 18 302 | **99.40%** |
| 4 | 67 108 864 | 1 426 648 | **14.470** | 14.486 | **3.618** | 22 704 | 1 359 540 | 1 426 648 | **58.85%** |
| 8 | 33 554 432 | 9 388 754 | **20.218** | 20.420 | **2.527** | 1 233 000 | 9 355 200 | 9 388 754 | **16.07%** |
| 16 | 16 777 216 | 13 999 353 | **23.373** | 23.975 | **1.461** | 11 100 000 | 13 982 576 | 13 999 353 | **2.54%** |

**Читать так:** H₈ = 20.2 бит на 8-байтовый блок = 2.53 бит/байт. Это *нижняя* оценка: истинная энтропия блока не ниже, потому что plug-in смещён вниз. xz даёт 1.94 бит/байт на всей последовательности, т.е. асимптотическая скорость ≤1.94, что согласуется с H₁₆/16 = 1.46 < 1.94 (H_k/k убывает с k).

### 3.3 Покрытие кодбука: сколько блоков реально покрывает V кодов

| V | k=2 | k=4 | k=8 | k=16 |
|---:|---:|---:|---:|---:|
| 256 | 76.21% | 20.21% | 4.72% | 0.93% |
| 1 024 | 94.08% | 36.41% | 8.91% | 1.49% |
| **4 096** | **99.40%** | **58.85%** | **16.07%** | **2.54%** |
| 8 192 | 99.90% | 69.60% | 20.97% | 3.33% |
| 65 536 | 100% | 89.84% | 40.20% | 7.49% |
| 1 048 576 | 100% | 99.44% | 69.71% | 22.81% |
| 4 194 304 | 100% | 100% | 84.52% | 41.56% |
| 16 777 216 | 100% | 100% | 100% | 100% |

**Это и есть ответ на «кодбук 4096 для 8 байт».** При k=8 кодбук V=4096 точно кодирует 16.07% позиций; остальные 83.93% — либо ошибка, либо escape. Для 99.9% покрытия нужно 9.36M кодов; для 100% — 9.39M на этой выборке (истинная поддержка больше). V=8192 покрывает 20.97%, V=65 536 — 40.20%. **Фиксированный кодбук 4096 на 8-байтовый патч lossy по построению, а не из-за слепоты рецептивного поля.**

### 3.4 Минимальный кодбук на длину патча: два разных определения

| k | H_k (бит) | Rate bound 2^H_k (бит) | V для 99.9% покрытия | V для 100% выборки | V=4096 покрывает |
|---:|---:|---:|---:|---:|---:|
| 2 | 9.03 | 523 | 8 136 | 18 302 | 99.40% |
| 4 | 14.47 | 22 704 | 1 359 540 | 1 426 648 | 58.85% |
| 8 | 20.22 | 1 233 000 | 9 355 200 | 9 388 754 | 16.07% |
| 16 | 23.37 | 11 100 000 | 13 982 576 | 13 999 353 | 2.54% |

**Два определения «lossless-in-expectation», и их нельзя смешивать.**

1. **По скорости (Shannon):** lossless-код тратит ≥ H_k бит/блок в среднем; фиксированный V-код тратит log₂V бит, поэтому `V ≥ 2^H_k` — **необходимое** условие. Для k=8 это 1.23M кодов; для k=16 — 11.1M.
2. **По ошибке (fixed codebook):** ожидаемая ошибка = 0 требует, чтобы каждый блок с ненулевой вероятностью был в кодбуке, т.е. `V ≥ |support|`. Для k=8 это ≥9.39M (на выборке), для k=16 ≥14.0M. **Это в 7.6× и 1.26× больше 2^H_k; для k=4 — в 60×, для k=2 — в 15.6×.** Причина — длинный хвост: 2^H_k покрывает только «типичное множество», а не поддержку.

**Следствие:** V=4096 при k=8 не удовлетворяет ни одному определению: он в 300× меньше rate bound 2^H_8 = 1.23M (по числу кодов), и в 2290× меньше support bound 9.39M. V=4096 при k=2 почти достаточен (99.40%), при k=4 — нет (58.85%).

### 3.5 Удовлетворяет ли какая-нибудь опубликованная latent-text схема этому при 95M?

| Схема | Что это | V / rate | Lossless? | При 95M? |
|---|---|---|---|---|
| **NCM** | 32-уровневый residual VQ, ℓ_K=L | 11 440 кодов × 12 бит / 1024 токена = **134 бит/токен** (OWT); 641 × 13 / 128 = **65 бит/токен** (LM1B) | 98.79–99.99% exact-match, **не lossless**; rate выше H, поэтому «да» по скорости и «нет» по ошибке | Нет: токенизатор 111M > генератора 94.65M; код в 5–25× энтропии |
| **BLT** | entropy patches + local decoder | Не фиксированный V: continuous latent + byte decoder | Локальный декодер может восстановить любой байт; **не кодбук** | Нет: 400M–8B; ps=8 хуже при 1B |
| **H-Net** | learned 4.8–5.3 B chunks | continuous chunker + decoder | Не фиксированный V | Нет: 760M/1.3B |
| **ByteFlow** | coding-rate chunks 2.0–5.12× | continuous latent + local dec/enc | Не фиксированный V | Нет: 600M/1.3B |
| **ConceptLM** | VQ над hidden states | BPE на входе; VQ не текстовый кодек | Нет | 70M–1.5B, но это не byte-кодек |
| **LCM / DLCM** | sentence/concept embeddings | continuous / variable-length | Нет | 1.3B+ |

**Ответ:** **нет.** Ни одна опубликованная latent-text схема при 95M не удовлетворяет lossless-границе для k=8 с малым V. Единственная дискретная схема в диапазоне 88–120M — NCM — удовлетворяет *rate* bound только потому, что тратит 65–145 бит/токен, т.е. **избыточна в 5–25× энтропии текста**; это не компрессия, а overcomplete-код. BLT/H-Net/ByteFlow обходят кодбук-границу, используя continuous latent + локальный декодер, но платят его FLOPs (5–73% шага) и не имеют 95M-результата.

---

## 4. Что должен делать кодек, чтобы выиграть

### 4.1 Формула

Пусть:
- `B` — базовая цена байта (MFLOP/byte) у dense 95M: **691.4** (block 567.4 + attention 122.7 + head 1.3) [C, `docs/reviews/2026-11-07-metrics.md` §4].
- `s` — цена одной позиции стека (MFLOP/position); при той же модели `s = B` (одна позиция = один байт в базе).
- `k` — байт на патч.
- `c` — цена кодека (MFLOP/byte).
- `f` — доля шага, не масштабирующаяся с числом позиций (optimizer, launches, host gap); измерено **f=0.64** при 314M [M, `2026-11-03` §3]; при 150M f≈0.87 [M, `~/logs/ab_bf400/*.jsonl`].

Тогда цена байта = `s/k + c`, а ускорение по байтам:

```
S = B / (s/k + c) = 1 / (1/k + c/B)          (если s = B)
```

С учётом фиксированной доли:

```
S_A = (1 − c) / (f + (1 − f)/k)              (равные байты на шаг)
S_B = k / (1 + k·c/B)                        (равные позиции на шаг, байтов ×k)
```

### 4.2 Два режима и что из них следует

| Режим | Формула | k=8, f=0.64 | k=8, f=0.15 | k=8, равные позиции |
|---|---|---:|---:|---:|
| Равные байты/шаг | `S_A = (1−c)/(f+(1−f)/k)` | потолок **1.46×** при c=0; для 3× нужно **c ≤ −1.055** — невозможно | для 3× нужно **c ≤ 0.231** (23.1%) | — |
| Равные позиции/шаг | `S_B = 8/(1+8c/B)` | **c ≤ 0.2083·B = 144.0 MFLOP/byte** | то же | **c_step/main_step ≤ 1.667** |

**Главный арифметический результат:** при измеренной f=0.64 **3× при равных байтах невозможно ни с каким кодеком**; потолок 1.46×. Для 3× в этом режиме нужно **f ≤ 0.238** даже при бесплатном кодеке. Это не проблема кодека — это проблема launch-bound шага.

В режиме равных позиций (8× байтов на шаг, то же VRAM, то же время шага) кодек может стоить **до 20.8% базовой цены байта = 144 MFLOP/byte**, или **167% шага основной модели**. Наш измеренный conv-кодек — 2.14 MFLOP/byte (2.9% FLOPs, 5.4% wall-clock), т.е. **в 67× ниже FLOPs-бюджета** и в 31× ниже wall-clock-бюджета.

### 4.3 Сравнение с измеренным и с causal-conv

| Кодек | MFLOP/byte | Доля B=691.4 | Доля от 144 | Комментарий |
|---|---:|---:|---:|---|
| Наш conv, 8 B/patch, 357k | **2.14** | 0.31% | 1.49% | 5.4% wall-clock на launch-bound шаге |
| Causal-conv, d_local=128, f=384, 2 слоя, kernel 5, маскированный вход | **0.25** | 0.036% | 0.17% | [C] 8·128·5 + 2·(4·128²+2·128·384) MAC/patch |
| BLT-like, 1B, p=8 | ~125M params, ~50% FLOPs | 50% | 240× | [C] из Table 10 |
| MegaByte-like, p=8 | 73.4% FLOPs | 73% | 350× | [C] |

**Вывод:** causal-conv кодек с маскированным входом стоит ~0.25 MFLOP/byte — в 576× меньше бюджета 144. Даже если он будет в 10× дороже из-за нескольких слоёв и более широкого receptive field, он остаётся в 50× ниже бюджета. **1.5 BPB потеря не может быть объяснена ценой кодека.** Она объясняется либо потерей информации (кодбук/латент не содержит патч), либо слепотой/оптимизацией (латент содержит, но модель не может им воспользоваться). Это и есть E1.

### 4.4 Что означает 5.4%

5.4% — это **wall-clock** доля на шаге 314M, где f=0.64 и шаг launch-bound. По FLOPs это 2.9%. Для 3× в режиме равных позиций бюджет 167% шага; 5.4% — это 3.2% от бюджета. Для 3× в режиме равных байтов бюджет отрицательный: 5.4% не спасает, потому что потолок 1.46×. **Кодек не является узким местом ни в одном режиме; узкое место — качество на байт и фиксированная доля шага.**

---

## 5. Три эксперимента, ранжированные

Протокол для всех: dense 95M (`port/configs/flagship3_dense.toml`), pure CE, `eval_batches` printed, same data order, 3 seeds, held-out window ≥40 960 B, win only if the delta beats the control's own seed spread (AGENTS §1.2). Costs at **60 kB/s = 4.63 h/GB**; 200 MB = 0.93 h; patch arm at 8× bytes/step = 0.42–0.44 h per 200 MB.

### E1 — Triage: слепота или потеря информации? (ранг 1; 6–8 GPU-h)

**Вопрос:** 1.5 BPB — это потому, что латент не содержит патч (information loss), или потому, что содержит, но модель не может им воспользоваться (blindness/optimization)?

| Плечо | Encoder видит | Decoder предсказывает | Что проверяет |
|---|---|---|---|
| C0 | — | байт t | контроль |
| A′ | байты < патча j (маска) | байты патча j | слепота: латент из прошлого патча |
| B | весь патч j | байты патча **j+1** | сдвиг на патч: информация есть, утечки нет |
| Probe | — | reconstruction патча из латента | exact-match, codebook usage, residual |

**Config change:** в `encode_chunks`/patch head — сдвиг латента на один патч (MegaByte Eq. 2–4); маска на свои байты для A′; gate `bytes[:, q:] → logits[:, :q]` на CPU; eval-only probe: encode 8-байтовый патч, decode, exact-match. Никаких новых параметров.

**Ожидаемая дельта vs C0 при равных байтах:** A′ **+0.2…+0.8 BPB** [S]; B **+0.05…+0.20 BPB** [S]. Если B в пределах seed spread C0 — слепота была причиной, сдвиг её лечит. Если B всё ещё >0.2 и reconstruction <90% — потеря информации. Если B >0.2, но reconstruction >99% — проблема оптимизации/архитектуры, не кодека.

**GPU-hours:** 3 arms × 3 seeds × 200 MB: C0 0.93 h, A′ 0.44 h, B 0.44 h → 1.81 h/seed × 3 = **5.4 h**; probe +0.5–1 h → **6–8 h**.

**Falsifier:** B ниже C0 на >0.3 BPB — утечка (не победа); B хуже C0 на >0.2 при reconstruction >99% — латент не используется; reconstruction <90% — кодек lossy, и никакое обучение это не исправит.

**Paper, предсказывающая знак:** Scratchpad §3.2 (patch lag) предсказывает, что A′ теряет; MegaByte §3.2 / BLT §3.3 («decode the next patch») предсказывают, что B восстанавливает; наш §3.3 предсказывает information floor.

### E2 — Equal-GPU-hours контроль (ранг 2; 27.8 GPU-h)

**Вопрос:** патчи покупают качество или только скорость? Потратить те же часы на **больше байтов**, а не на меньше позиций.

| Плечо | Единица | Байтов за 4.63 h | Шагов | Ожидаемый BPB | Основание |
|---|---|---:|---:|---|---|
| C0 | 1 B/pos | 1.0 GB | 40 690 | baseline | measured slope 98.3→268 MB = −0.408 BPB [M, `2026-11-07-architecture-frontier.md`] |
| P8 | 8 B/patch | **8.0 GB** | 40 690 | **−0.6…+0.2 vs C0** [S] | 8× data at 84 B/param vs 10.5; per-byte penalty 0.05–0.15 [T, 2412.09871 §5.1] |

**Config change:** `use_byteflow=true`, `byteflow_codec=attention` (causal, already measured 3.2427), `byteflow_k_tokens` so mean patch = 8 B; byte arm on 1 GB, patch arm on 8 GB, same wall clock, 3 seeds.

**GPU-hours:** 3 × (4.63 + 4.63) = **27.8 h**; screening at 500 MB/4 GB = **13.9 h**.

**Falsifier:** P8 хуже C0 при 8 GB → патчинг — speed-only lever, не база рецепта. P8 лучше на >0.1 BPB → data-efficiency thesis держится в data-starved режиме; повторить при 22 GB, где знак должен перевернуться.

**Paper:** BLT §5.1 (ps=8 хуже при 1B), `2610.05978v1` §3.2 (subword лучше при равных FLOPs), MegaByte Table 2 (quality at fixed compute, 3.2× params).

### E3 — Byte-bag / TST (ранг 3; 5.6 GPU-h)

**Вопрос:** единица «bag из 4 байт» без кодека даёт что-нибудь при 95M? Это самый дешёвый положительный prior (TST 2.5× на 10B, но token-level).

| Плечо | Единица | Loss | Ожидаемая дельта | Основание |
|---|---|---|---|---|
| C0 | 1 B/pos | CE | baseline | — |
| B4 | bag из 4 байт | multi-hot CE | **−0.05…−0.15 BPB** при равных шагах [S] | TST 270M 3.212→3.142 (−0.070); 3B 2.808→2.676 (−0.132) [T, Table 1] |

**Config change:** bag size 4 (байты, не токены), multi-hot CE по 256 байтам, 30% обучения в superposition, затем обычный NTP recovery; без нового кодека. 2 arms × 3 seeds × 200 MB.

**GPU-hours:** 2 × 3 × 0.93 = **5.6 h**.

**Falsifier:** нет выигрыша при равных шагах, или recovery съедает выигрыш → единица не рычаг при byte-level; тогда остаётся только кодек/архитектура.

**Paper:** TST `2605.06546v2` Table 1 (270M −0.070; 600M −0.076; 3B −0.132; 10B 2.5× time).

### Ранжирование (RU)

| Ранг | Эксперимент | GPU-h | Почему этот ранг |
|---|---|---:|---|
| **1** | **E1** triage | 6–8 | Решает, имеет ли смысл вся кодек-линия; дешевле E2 в 4×; отделяет information loss от blindness |
| **2** | **E2** equal-GPU-hours | 27.8 (screen 13.9) | Единственный тест метрики владельца — BPB на (GiB×час); но бессмыслен, пока E1 не показал, что кодек не lossy |
| **3** | **E3** byte-bag/TST | 5.6 | Самый дешёвый положительный prior; проверяет саму единицу без кодека; запускать, если E1 покажет information loss |

**Если запускать только один:** E1. Если E1 показывает information loss — E3 (единица без кодека) и закрыть кодек-линию. Если E1 показывает blindness — E2 (сдвиг уже в E1, дальше equal-GPU-hours).

---

## 6. VERIFIED (sourced) vs SPECULATION (my inference)

### VERIFIED (sourced)

- **NCM:** `2610.14162v1`, 2026-10-09, v1 only. Table 5 (K=8–32, V=16–8192, d=384, enc 0, dec 1–4, tokenizer 14.90–144.00M), Table 6 (generator 25.46–119.96M), Table 1 (99.99/99.86/99.98/99.79/98.79/100), Table 2 (AR 99M 25.2 BLEU at 64 NFE; NCM 120M 22.6 at 11; NCM+MBR 25.2 at 132), Table 3 (Sudoku 99.5/96.5/86.1 at 12 NFE), §4.6 (21×/10×/31× vs DUO/OWT/AURORA-LM, H100, 1024 seq, batch 16). Code `github.com/HazyResearch/ncm` main, pushed 2026-10-14; `config/eval/owt.yaml` levels 1..1024, `config/eval/lm1b-small.yaml` levels 1..128; `models/vqvae.py` “enc_layers=0 / dec_layers=1”.
- **BLT:** `2412.09871v1`, 2024-12-13; Table 10 (1B: enc 12M, global 1B, dec 113M), §5.1 (“ps=8 starts significantly worse at 1B, better at 7B”), §5.3 (crossover 2.5–3× compute-optimal).
- **MegaByte:** `2305.07185v2`, 2023-05-19; Table 2 (1.057→1.000 PG-19; 0.575→0.411 code; 0.816→0.678 arXiv), §3.2 (shifted patch input).
- **SpaceByte:** `2404.14408v3`, 2024-10-06; Table 1 (1.112/0.804/0.552), §4 (“roughly matches tokenized”), BLT §5.1 (“remains far from Llama 3”).
- **H-Net:** `2507.07955v2`, 2025-07-15; Table 2 (760M 0.756→0.743 at 0.42→0.43 GFLOP/byte; 1.3B 0.730→0.715 at 0.69).
- **Scratchpad:** `2605.09630v1`, 2026-05-10; Table 1 (byte 54.1; p=16 48.0/5.7×; p=16+SP 54.2/2.9×; p=8+SP 53.8/2.6×), §3.2 patch lag, ~2B params/400B bytes.
- **TST:** `2605.06546v2`, 2026-05-19; Table 1 (270M 3.212→3.142; 600M 3.019→2.943; 3B 2.808→2.676; 10B 2.252→2.236; 2.5× at 10B A1B).
- **Byte-LM:** `2610.05978v1`, 2026-10-05; §3.2 (“under the same FLOPs budgets, Subword is always better”; byte 3.2× bytes/param), Table 1.
- **ConceptLM:** `2602.08984v1`, 2026-02-09; Table 1 (GPT2-124M 101.94→84.75; Pythia-70M +3.6; 410M +0.8), §6.5 (−11.3% train time).
- **LCM:** `2412.08821v2`, 2024-12-15; Table 3, Table 4 (smaLlama 34.88 vs LCM 33.64 ROUGE-L).
- **DLCM:** `2512.24617v2`, 2026-01-05; Table 2 (43.92 vs 41.23, +2.69), §7.1 (2.3B vs 1.3B; matched inference FLOPs).
- **ByteFlow:** `2603.03583v1`, 2026-03-03; Table 4 (0.6B: 1.02e21 FLOPs/0.89 BPB LLaMA vs 1.07e21/0.86 ByteFlow), §5.3 (+1.74 at 600M; +3.04 at 1.3B).
- **Proxy:** `2602.04289v2`, 2026-02-04; §3 (gzip fails; tokenizer/neural proxies transfer; gains grow with scale; 0.5B–14B).
- **Lossy Horizon:** `2510.22207v1`, 2025-10-25; Table 1 (EPC 1.1–1.3 BPC at 0.98 CharFid; PM 2.3–2.8 BPC), VQ+RE dominated.
- **Our corpus measurement:** 8 × 32 MiB, H₁–H₁₆, coverage, xz 1.940 b/B; offsets and estimator in §3.1; reproducible with `xz -9` and numpy.
- **Repo audit:** `~/logs/ab_bf400/{bf8_s1,full_s1}.log` (4.135 vs 2.684; 150.8M vs 199.3M); `~/logs/night2/ab_byteflow.log:133` (3.6152 vs 3.2427 at 1.9M); `docs/reviews/2026-11-03-three-verdicts-and-a-leak.md` §3 (47.6→32.6 ms, 5.4%, 0.4992 leak).

### SPECULATION (my inference, no direct benchmark)

- **S1 — BPB floor.** 0.0003–0.038 BPB assumes 10–15 bits per substitution (12.5 midpoint), 4–4.5 B/token, independent errors. The paper reports exact-match, not NLL; the decoder NLL could be larger. Marked [S] in §1.3.
- **S2 — 95M transfer.** ByteFlow 600M (+5% FLOPs, −0.02…−0.03 BPB), H-Net 760M (−0.013), Scratchpad 2B (2.9×), TST 270M (−0.070) are all at ≥270M. No 95M measurement exists; the transfer is an assumption.
- **S3 — E1 deltas.** A′ +0.2…+0.8, B +0.05…+0.20 at equal bytes. These are transfers of BLT’s ps=8-at-1B and Scratchpad’s patch-lag, not measurements at 95M.
- **S4 — E2/E3 deltas.** −0.6…+0.2 BPB (E2) and −0.05…−0.15 BPB (E3) are hypotheses, not paper numbers.
- **S5 — The 1.5 BPB attribution.** Until E1 runs, “blindness” and “information loss” are both consistent with the artifacts; the confound (150.8M vs 199.3M, missing Engram/mHC/CSA/LayerRoPE) is the only established fact.
- **S6 — The 2.04× claim.** No artifact prints it; the report treats it as unverified, not as a measurement.

**Confidence.** **HIGH** on: NCM v1/date, codec/generator params, reconstruction, speed baselines, absence of AR/byte comparison, the code rate from the released config, the H_k/coverage measurement, the codec arithmetic, ByteFlow Table 4, H-Net Table 2, Scratchpad Table 1, TST Table 1. **MEDIUM** on: the transfer of ByteFlow/H-Net/Scratchpad/TST to 95M; the exact ℓ_k of the paper’s main runs (the released config may be a different checkpoint: V=8192 vs 4096, 16 layers vs 12). **LOW** on: every expected delta in §5 (S3–S4) and the BPB floor (S1). The confidence rating applies only to the VERIFIED part.

---

## 7. Open questions

1. **Does the released NCM checkpoint reproduce the Gen. PPL, and what is its BPB on our eval tail?** The checkpoint is on HuggingFace (`lm1b/small`, `lm1b/medium`, `lm1b/large`, `owt`); a CPU/GPU scoring run would settle whether the code matches the paper. This is the cheapest external check.
2. **What is the exact ℓ_k for the paper’s main runs?** Table 5 omits it; the released config gives 1..1024 and 1..128, but with V=8192 and 16/24 layers, not the paper’s 12/768 and V=4096. The code rate in §1.2 assumes the config’s schedule; if the paper used a different schedule, the rate changes by 2×.
3. **Does the 1.5 BPB loss survive a clean equal-param A/B?** E1 answers this; the current artifact is confounded.
4. **Does ByteFlow’s +5% FLOPs/−0.02…−0.03 BPB at 600M survive at 95M/1 GB?** No data; E2/E3.
5. **Is the codec’s wall-clock share 5.4% at 95M, or larger?** At 314M f=0.64; at 95M the fixed share is larger, so the codec’s relative cost may be higher, but the FLOPs budget (144 MFLOP/byte) is unchanged. This matters for E2’s screening cost.
6. **Can a causal-conv codec with a patch-masked input reach the 144 MFLOP/byte budget and recover the 1.5 BPB?** The arithmetic says yes on cost; the quality is the open question, and E1’s B arm is the first test.

---

## Sources

**Full text [T] / abstract [A] / code [C] / repo [M] / computed [C]:**

1. **NCM** — Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, arXiv:2610.14162v1, 2026-10-09. Tables 1–3, 5–7; §4.6; App. C.1–C.2, G, I.
2. **NCM code** — `github.com/HazyResearch/ncm` @ `main`, pushed 2026-10-14. `config/eval/owt.yaml`, `config/eval/lm1b-small.yaml`, `models/vqvae.py`, `models/next_concept.py`.
3. **NCM blog** — `hazyresearch.stanford.edu/blog/2026-10-12-ncm` (2026-10-12). [A]
4. **BLT** — Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens*, arXiv:2412.09871v1, 2024-12-13. Table 10; §5.1, §5.3.
5. **MegaByte** — Yu et al., *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*, arXiv:2305.07185v2, 2023-05-19. Table 2; §3.2.
6. **SpaceByte** — Slagle, *SpaceByte: Towards Deleting Tokenization from Large Language Modeling*, arXiv:2404.14408v3, 2024-10-06. Table 1; §4.
7. **H-Net** — Hwang, Wang, Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, arXiv:2507.07955v2, 2025-07-15. Table 2.
8. **Scratchpad** — Zheng et al., *Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models*, arXiv:2605.09630v1, 2026-05-10. Table 1; §3.2, §4.2.
9. **TST** — Peng, Gigant, Quesnelle, *Efficient Pre-Training with Token Superposition*, arXiv:2605.06546v2, 2026-05-19. Table 1.
10. **Byte-LM scaling** — Wang et al., *Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation*, arXiv:2610.05978v1, 2026-10-05. §3.2; Table 1.
11. **ConceptLM** — Liu et al., *Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models*, arXiv:2602.08984v1, 2026-02-09. Tables 1–2; §6.5.
12. **LCM** — LCM team, *Large Concept Models: Language Modeling in a Sentence Representation Space*, arXiv:2412.08821v2, 2024-12-15. Tables 3–4.
13. **DLCM** — Qu et al., *Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space*, arXiv:2512.24617v2, 2026-01-05. Table 2; §7.1.
14. **Coconut** — Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space*, arXiv:2412.06769v4, 2026-08-23. [A]
15. **ByteFlow** — Deng et al., *ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer*, arXiv:2603.03583v1, 2026-03-03. Table 4; §5.3.
16. **Proxy Compression** — *Proxy Compression for Language Modeling*, arXiv:2602.04289v2, 2026-02-04. §3.
17. **Lossy Horizon** — *The Lossy Horizon: Error-Bounded Predictive Coding for Lossy Text Compression (Episode I)*, arXiv:2510.22207v1, 2025-10-25. Table 1.
18. **Repo prior reads** — `docs/research/2026-10-21-hybrid-ncm.md` (NCM first read), `docs/research/2026-11-07-patch-compute.md` (patch landscape and arithmetic), `docs/reviews/2026-11-03-three-verdicts-and-a-leak.md` §3 (codec share, leak, 47.6→32.6 ms), `docs/reviews/2026-11-07-metrics.md` §4 (FLOPs ledger), `.bulba/goal.md` (owner metric).
19. **Logs** — `~/logs/ab_bf400/bf8_s1.log`, `~/logs/ab_bf400/full_s1.log`, `~/logs/night2/ab_byteflow.log:133`, `~/logs/ab_bf400/*.jsonl`.
20. **This report’s corpus measurement** — 8 × 32 MiB of `real_filtered/corpus.bin`, offsets printed in §3.1; H₁–H₁₆, coverage, `xz -9` = 1.940 b/B. Reproducible with the script in `/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/research_tmp/hk.py`.
