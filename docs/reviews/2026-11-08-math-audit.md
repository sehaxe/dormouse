# 2026-11-08 — Аудит математики Python-лейна

Read-only аудит формул `port/src/dormouse/` против статей, на которые они ссылаются.
Код не менялся; единственный записанный файл — этот отчёт. GPU не трогался
(`CUDA_VISIBLE_DEVICES=""`), все проверки — CPU, torch 2.13.0+cu129.

**Теги:** `[T]` — слова статьи/референса; `[C]` — наша арифметика (измерено здесь);
`[S]` — спекуляция. Числа без тега — замеры этого аудита.

**Что реально прогнано:**

```
pytest tests/test_muon_shape_fair.py tests/test_muon_reference.py       7 passed
pytest tests/test_segment_csa.py                                       13 passed
pytest tests/test_long4m.py tests/test_ce_fixed.py tests/test_model_internals_audit.py
                                                                       passed
pytest tests/test_optimizer_suite.py -k "ema or wsd or z_loss or softcap or adamw"
                                                                       13 passed
pytest tests/test_lr_tail.py                                           1 FAILED
    test_min_lr_frac_outside_its_range_is_refused[0.0] — конфиг больше не
    отказывает min_lr_frac=0.0, тест ждёт LoudError. Код и тест противоречат
    друг другу на HEAD (см. §5.3).
```

Числовые скрипты лежат в `/tmp/opencode/mathaudit_*.py`; чекпойнты для «настоящего»
импульса — `checkpoints/flagship/flagship.pt` (step 149 999),
`checkpoints/flagship2/flagship2.pt` (74 999), `checkpoints/dense103/dense103.pt`
(34 999). Это единственные реальные momentum-буферы, до которых можно дотянуться
без GPU.

**Короткий вердикт.** Арифметика шага Muon, носимых норм, CE, WSD (на нормальном
домене), EMA и KDA — в порядке. Два места действительно сломаны по-крупному:
**иерархический CSA** (падает на `topk < n_parent` и имеет утечку parent-маски,
из-за которой `forward` и `forward_state` — разные функции), и
**документированное утверждение про «1.17e-15 от многообразия»** у
Newton–Schulz, которое ложно на 14 порядков и на котором держится решение
3-vs-5 шагов. Плюс граница WSD (`warmup + decay > total`), абсолютный ε в
NS-прескейле для мелких/ill-conditioned буферов и красный тест на
`min_lr_frac=0`.

---

## 1. Muon (`train/optim.py`)

### 1.1 Что заявлено и что считается

| # | formula | code | source | verdict |
|---|---|---|---|---|
| M1 | `Ortho(X) = U Vᵀ` (точный polar) | `zeropower_via_ns`, `optim.py:256-301` | Muon+ Eq. (4), §3.1: `O_t = Norm_d(Ortho(M_t))` | **WRONG as claimed** — код реализует не `Ortho`, а `U S' Vᵀ`; см. 1.2 |
| M2 | `M_t = μ M_{t-1} + (1−μ) G_t` | `optim.py:818-855`, `:1434` | Muon+ Eq. (4), без Nesterov | **RIGHT** |
| M3 | `(a,b,c) = (3.4445, −4.7750, 2.0315)`, `X ← aX + (bA + cA²)X`, `A=XXᵀ` | `optim.py:87`, `:270-280`, `:335-380` | Muon+ App. D.1, который сам цитирует Jordan; статья коэффициентов не выводит | **RIGHT** (литерал совпадает; цитата поддерживает только коэффициенты, не «polar») |
| M4 | `O = Norm_row(Norm_col(X))`: сначала столбцы, потом строки | `colrow_normalize`, `optim.py:457-491` | Muon+ Eq. (7), Alg. 1 | **RIGHT** по порядку; ε стоит не там, где в статье (benign, §1.5) |
| M5 | `W ← W − η·√c·O` (shape-fair) | `_muon_scale`, `optim.py:1122-1136`; `lr_mode` | — (это наша правка, не статья) | **RIGHT**: `RMS(ΔW) = lr` точно |
| M6 | `x ← x / (‖x‖_F + 1e-7)` | `optim.py:272`, `:341` | Jordan `muon.py` | **WRONG** как инвариант: ε абсолютный, не относительный; при `‖m‖ ≪ 1e-7` цепочка линейна |
| M7 | «3 шага — ничья на rank 64; 5→3 стоит +0.27 BPB на `[832,3072]`» | `docs/architecture/2026-11-08-the-16x.md:33-52` | `2606.00371` §3.2 (ничья), наш A/B (0.27) | **WRONG** в объяснении; измерение 0.27 правдоподобно, но причина не форма, а спектр (1.4) |

### 1.2 Newton–Schulz — это не polar, и это не баг кода

Референс Keller Jordan (`/tmp/opencode/muon_upstream.py`, скачан 2026-11-08,
286 строк) сам пишет:

> `[T]` "This iteration therefore does not produce UV^T but rather something like
> US'V^T where S' is diagonal with S_ii' ~ Uniform(0.5, 1.5), which turns out not
> to hurt model performance at all relative to UV^T."

Код порта — точная транскрипция этого референса. Но тогда все утверждения вида
«polar step», «on the Stiefel manifold», «`Ortho(M)`» — это утверждения о другом
алгоритме. Скалярная карта `p(σ)=aσ+bσ³+cσ⁵` имеет неподвижные точки
`σ=0.868` и `σ=1.264`, обе **отталкивающие**; устойчив 2-цикл. Прямой счёт
(диагональный вход, `[C]`):

| k | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| `p^k(0.5)` | 0.701001 | 1.113620 | 0.720705 | 1.089975 | 0.696437 | 1.081025 |

То есть после 5 шагов сингулярные числа не «1», а 0.696/1.081 — **двухтактный
цикл**, а не сходимость. Это ровно то, что обещает upstream; это не ошибка
транскрипции.

**Ложное утверждение.** `docs/decisions/2026-10-15-1b-readiness-plan.md:280`,
`docs/research/2026-10-16-muon-orthogonalization-cost.md:69` и
`docs/architecture/2026-11-08-the-16x.md:50` повторяют: «NS-3 сидит 1.17e-15 от
многообразия». Измерено на фикстуре из `test_train.py` (64×8, spectrum 10:1,
`zeropower_via_ns`, k=3):

```
orth_per_entry = ‖O Oᵀ − I‖_F / rank = 1.732e-01     (1.17e-15 заявлено)
‖O − polar(X)‖_F / ‖polar(X)‖_F     = 3.135e-01
```

Ни при одном k=1..8 это не приближается к 1e-15: цепочка не сходится к
многообразию вообще. Заявленное число, вероятно, относится к другому алгоритму
(QR/PolarExpress-ретракция), а не к `zeropower_via_ns`. Это надо править в трёх
документах и в docstring `tests/test_train.py:61`; сам тест это число не
проверяет, поэтому зелёный.

**Числовая проверка, которая ловит ложь:** диагональный тест выше — `p^k(0.5)`
для k=1..5; или `_ortho_per_entry` на `randn(64,8) @ diag(logspace(0,-2,8))` при
k=3 (получается 1.7e-1, не 1e-15). Оба — CPU, секунды.

### 1.3 Что на самом деле делают 3 vs 5 шагов (реконсиляция +0.27)

Ключевой факт: `colrow_normalize` фиксирует **масштаб** шага, поэтому 3-vs-5 —
это **только перевзвешивание сингулярных направлений**, никогда не размер шага.
Измерено (RMS(ΔW)/lr после ColRow, `[C]`):

| shape | k=1 | k=3 | k=5 | k=8 | shape_fair RMS/lr (любой k) | uniform RMS/lr |
|---|---:|---:|---:|---:|---:|---:|
| `[832,3072]` randn | 1.804e-2 | 1.804e-2 | 1.804e-2 | 1.804e-2 | **1.000000** | 1.804e-2 |
| `[3072,832]` randn | 6.662e-2 | 6.662e-2 | 6.662e-2 | 6.662e-2 | **1.000000** | 6.662e-2 |
| `[832,832]` randn | 3.467e-2 | 3.467e-2 | 3.467e-2 | 3.467e-2 | **1.000000** | 3.467e-2 |
| `[2048,64]` randn | 0.7071 | 0.7071 | 0.7071 | 0.7071 | **0.999999** | 0.7071 |

То есть `_muon_scale` — **RIGHT**, ровно `RMS(ΔW)=lr`; старый `uniform` даёт
дефициты 55.4× / 15.0× / 28.8× / 1.41×, что совпадает с аудитом 2026-11-07.

Теперь направление. `relpolar = ‖NS_k − polar‖/‖polar‖` и
`cos(ColRow NS3, ColRow NS5)`:

| вход | cond | relpolar k=3 | relpolar k=5 | cos(NS3,NS5) | вывод |
|---|---:|---:|---:|---:|---|
| `[832,3072]` randn | 3.15 | 1.57e-1 | 1.66e-1 | **0.9819** | **ничья** |
| `[3072,832]` randn | 3.14 | 1.57e-1 | 1.66e-1 | **0.9818** | **ничья** |
| `[2048,64]` randn | 1.39 | 2.63e-1 | 2.71e-1 | **0.9973** | **ничья** |
| `[832,832]` randn | **1948** | **4.09e-1** | **2.07e-1** | **0.9143** | **не ничья** (квадрат: хвост cond) |
| `[832,3072]` spectrum 10:1 | 10 | 3.56e-1 | 1.90e-1 | — | 3 хуже в 1.9× |
| реальный `[1024,64]` (flagship) | 14.4 | 1.82e-1 | 1.92e-1 | **0.9850** | **ничья** |
| реальный `[64,2730]` (flagship2, norm 1e-4) | 27 | — | — | **~1.0000** (fp32 dot/norm) | **ничья** |
| реальный `[8192,1536]` (dense103, dense FFN) | **3660** | **7.80e-1** | **3.69e-1** | **0.7112** | **не ничья** |
| реальный `[1536,8192]` (dense103) | **1496** | **7.88e-1** | **3.40e-1** | **0.7076** | **не ничья** |

**Реконсиляция.** «3 шага — ничья на rank 64» верно для реальных rank-64
буферов (cos 0.985–1.0), но причина не `rank=64` как форма: случайная
`[832,3072]` (cond 3.15) — тоже ничья (relpolar 0.157 vs 0.166, cos 0.982),
а случайная `[832,832]` с cond **1948** (квадратная матрица имеет почти
нулевое наименьшее сингулярное) — уже нет (relpolar 0.409 vs 0.207,
cos 0.914). Причина — **спектр импульса**: у trained dense-матрицы он 10³–10⁴,
у низкоранговых факторов — 10–30. +0.27 BPB на `[832,3072]` правдоподобен
именно потому, что реальный dense-импульс имеет cond 10³–10⁴ и
`cos(NS3,NS5)=0.71`: это на 76 % другой апдейт каждый шаг. Формулировка
16×-документа «на `[832,3072]` с плотным спектром 3-шаговая цепочка не
сходится» **неверна в посылке**: 5-шаговая тоже не сходится (relpolar 0.37);
разница — в перевзвешивании хвоста, а не в сходимости. Правильная
формулировка: *«3 шага сильнее подавляют хвост; на dense-импульсе это
относительное расхождение 76 % (cos 0.71), на реальных rank-64 — 1–7 %
(cos 0.985–0.997)»*.

**Cheapest fix:** не код, а гейт. `scripts/ns_step_ab.py` на сохранённых
momentum-буферах: для каждого shape-класса печатать `cos(ColRow NS3, NS5)`;
правило «3 шага — ничья» разрешать только при `cos > 0.98`. Сейчас такого гейта
нет; 16×-документ опирается на неверное 1.17e-15.

### 1.4 Абсолютный ε в прескейле — реальная дыра

`x = x / (x.norm() + 1e-7)` (`optim.py:272`, `:341`). Если `‖m‖ ≪ 1e-7`,
знаменатель — это `1e-7`, и `x_scaled ≈ 1e7·x`, т.е. **не** единичный по
Фробениусу. Тогда `p(σ)≈aσ` (линейный режим), и NS не ортогонализует вообще.

Живые чекпойнты (`[C]`):

| ckpt | shape | `‖m‖` | `sv_max/sv_min` на входе | `sv_max/sv_min` после NS-5 |
|---|---|---:|---:|---:|
| flagship2 | `[64,2730]` (256 шт.) | 1.03e-6 … 3.85e-4 | 27.7 | 1.66 (сжато) |
| flagship2 | `[2048,4]` | **1.44e-7** | **3.78e8** | **1.66e7** (не сжато) |
| dense103 | `[3072,4]` | **1.70e-7** | **2.40e8** | **4.70e7** (не сжато) |
| flagship (150k) | `[64,2730]` (256 шт.) | **0.0 … 5.9e-5** (median 6.8e-13; 62 буфера ровно 0.0) | 26.9 (sampled buffer, `‖m‖=9.2e-19`, `svmax=3.9e-19`) | 27.0 (линейный режим) |

Для `flagship`-буфера `svmax=3.9e-19` (`‖m‖=9.2e-19`) прескейл даёт `~9e-12`,
а `gram@gram ~ 6e-45` — **на полу субнормалей fp32** (min subnormal 1.4e-45):
член `c` исчезает/денормализуется, остаётся `a·x`, т.е.
`NS_5 ≈ 3.44^5 x` — не поляр, а масштабированный вход.
Часть буферов там ровно **0.0** (62 из 256; мёртвый импульс: NS даёт 0,
ColRow — 0, параметр не двигается). У ненулевых крошечных буферов после
ColRow строки снова единичные, поэтому параметр «не взрывается», но
ортогонализации нет. Тест `test_train.py::test_muon_update_compresses_the_spectrum`
на фикстуре 64×8 проходит; на живом `[2048,4]` он бы упал
(`sv ratio 1.66e7` против `< 2.0`).

**Cheapest fix:** относительный пол, `x / x.norm().clamp_min(torch.finfo(x.dtype).tiny)`
(или `x / (x.norm() + 1e-7 * x.norm().detach())`), и/или маршрутизировать
`min(shape) ≤ 4` в AdamW. Гейт: взять `opt['muon']['state'][k]['momentum']` из
`checkpoints/flagship2/flagship2.pt` для `[2048,4]`, прогнать NS-5, потребовать
`sv_max/sv_min < 2`; сейчас 1.66e7. CPU, секунды.

### 1.5 Момент и ColRow

Момент — `m ← μ·m + (1−μ)·g`, без Nesterov (`optim.py:846-848`), ровно Eq. (4)
Muon+. Проверка: первый шаг из нуля даёт `(1−μ)g`; NS scale-invariant, поэтому
`‖g‖·1e-6` и `‖g‖·1e3` дают **тот же** `RMS(ΔW)` (это уже измерено в аудите
2026-11-07; повторено на `[832,3072]`: 1.804e-2 при обоих масштабах). Если бы
`(1−μ)` не было, первый шаг был бы в 20× больше при μ=0.95 — гейт: один шаг,
сравнить с `(1−μ)·g`.

ColRow: порядок «столбцы, затем строки» совпадает с Eq. (7)
(`Norm_row(Norm_col(X))`). ε: статья `denom = sqrt(sum + eps)` (Alg. 1,
`docs/papers/muon-plus.md:96-115`); код `x / (norm + eps)`
(`optim.py:486-488`). При норме 1e-4 две формы расходятся в 3.3×, при 1e-8 —
в ~2900×, при норме O(1) — на 1e-7. Живой режим после NS — нормы O(1) у
полноранговых матриц, поэтому **benign** там; у `[2048,4]` с cond 1e8 — уже
нет, но там раньше срабатывает прескейл (§1.4). Это отклонение от литерала,
и его надо назвать, а не прятать.

### 1.6 Дефолты, которые врут

* `TrainConfig.muon_lr_mode = "uniform"` (`config.py:2057`), а
  `Muon(lr_mode="shape_fair")` (`optim.py:892`) и
  `MixOptimizer(muon_lr_mode="shape_fair")` (`optim.py:1727`). Тренер всегда
  передаёт `cfg.muon_lr_mode`, поэтому живые прогоны — `uniform`; но любой
  прямой `MixOptimizer(...)`/`Muon(...)` (тесты, скрипты, внешний код) молча
  получает шаг в 15–55× больше. Это не «shipped fix», а ловушка.
* `Muon(lr=0.02)` + `shape_fair` = `RMS(ΔW)=0.02` на элемент; опубликованный
  Muon с `lr=0.02` и `max(1,m/n)^0.5` даёт ~0.02/√c ≈ 7e-4 при c=832. Прямой
  пользователь `Muon` получает шаг в ~28× горячее опубликованного. Дефолт
  `lr_mode` надо либо сделать `uniform`, либо менять дефолт `lr`.
* Сообщение `ns_steps=3 = the measured-parity arm` (`optim.py:909`) устарело:
  16×-документ говорит +0.274 BPB, не parity.
* Docstring `Muon` (шапка файла, `optim.py:21`) описывает шаг
  `W ← W − η·max(1,m/n)^0.5·O_t`, т.е. `uniform`, а конструктор по умолчанию
  `lr_mode="shape_fair"`. Читатель docstring получает не тот шаг, который
  соберётся. Это та же болезнь, что и `TrainConfig` vs конструктор, но в
  документации.

---

## 2. Attention (`model/transformer.py`; GQA/softcap — в других файлах)

Бриф называет `transformer.py`, но в нём **нет GQA и нет logit softcap** — это
vanilla-контрол. GQA живёт в `model/kda.py`, softcap — в `losses.py`. Аудирую
и то, и другое; это не придирка к брифу, а факт о коде.

| # | formula | code | source | verdict |
|---|---|---|---|---|
| A1 | `θ_j = p·10000^(−2j/hd)`, split-half rotate | `rope_tables`/`apply_rope`, `transformer.py:60-83` | control's RoPE; NeoX/HF `rotate_half` | **RIGHT** |
| A2 | `scale = 1/√hd` | `transformer.py:153`, передан в SDPA `:261` | GPT-2 | **RIGHT** |
| A3 | causal mask | `is_causal=True`, `:261` | — | **RIGHT** |
| A4 | GQA kv-head → consecutive query heads | `kda.py:327-338`, `repeat_interleave` | Ainslie et al. 2023 | **RIGHT** (в `kda.py`, не в `transformer.py`) |
| A5 | `cap·tanh(z/cap)` | `losses.py:70-92`; вызовы `model.py:1108`, `loopblock.py:1421` | Gemma-2, 2408.00118 §2 | **RIGHT** |
| A6 | `d/dz cap·tanh(z/cap) = sech²(z/cap)` | — | — | **RIGHT**; в `z=cap` = **0.419974342** |

**Числа.**

* RoPE: против f64-референса `ang[p,j]=p·10000^(−2j/hd)` — freq `max|Δ|=3.40e-7`,
  rotation `max|Δ|=9.90e-7`; дрейф нормы `1.13e-8`. Применяется только к q и k
  (`transformer.py:260`), v не вращается — если бы вращался, rotation v сдвинул
  бы выход на O(1) (гейт: `apply_rope(v) != v`).
* SDPA: ручной `softmax((qkᵀ)/√hd)` с causal-маской совпадает до **4.44e-16**.
  Смена будущего key/value на позиции 6 не меняет выход позиций ≤3 **ровно
  0.0**; без causal было бы O(1).
* GQA: `n_kv_heads=2, n_heads=8`, `g=4`; выходные головы 0..3 — kv-голова 0,
  4..7 — kv-голова 1. Если бы стоял `repeat` вместо `repeat_interleave`,
  голова 1 читала бы kv-голову 1 вместо 0 — гейт на `[0,0,0,0,8,8,8,8]`.
* Softcap: `cap=15`: `y(0)=0`, `y(cap)=11.4239`, `y(3cap)=14.9258 < 15`;
  производная в `z=cap` = **0.419974342** (sech²(1)), в `z=3cap` = 0.009866.
  Это значит, что на «крышке» градиент CE ослаблен до 42 % — не ошибка, но
  нигде не записано; «CE margin ≤ 2·cap = 30» — верхняя граница, реальный
  максимум при `z=3cap` — 29.85.
* `transformer.py` LayerNorm: `nn.LayerNorm(d)` = eps **1e-5 внутри sqrt**;
  ручная формула `(x−μ)/√(σ²+1e-5)` совпадает до 4.44e-16. Внешняя форма для
  ε=1e-5 при unit-дисперсии отличается на ~5e-6 (для RMSNorm с ε=1e-3 —
  на 4.13e-3), поэтому гейт на перенос ε должен быть на RMSNorm, а не на
  LayerNorm с его дефолтным ε.

---

## 3. Нормировки и residual

| # | formula | code | verdict |
|---|---|---|---|
| N1 | RMSNorm `x/√(mean(x²)+ε)·γ` | `nn.RMSNorm(d, eps=cfg.norm_eps)` (`model.py:257,290,322,443`; `loopblock.py:360`) | **RIGHT** (ε внутри sqrt) |
| N2 | LayerNorm `(x−μ)/√(σ²+ε)·γ+β`, ε=1e-5 | `transformer.py:163-166` | **RIGHT**; но `cfg.norm_eps` до блоков **не доходит** |
| N3 | output-proj init `σ=0.02/√(2L)` | `transformer.py:193-196` | **RIGHT** |
| N4 | residual stream fp32 | `transformer.py:283`; `mm_mixed(...).to(x.dtype)` | **RIGHT** |

**Числа.** RMSNorm: ручная `x/√(mean(x²)+1e-3)` совпадает до **4.44e-16**;
«ε снаружи» отличается на **4.13e-3** — гейт на перенос ε. LayerNorm: 4.44e-16.
Init (L=12): `proj` std **0.004082715** против `0.02/√24 = 0.004082483`;
`fc2` **0.004085916**; `qkv`/`fc` **0.019992/0.020014**. Residual: вход/выход
блока fp32; bf16-блок на fp32-входе возвращает **fp32** (residual не понижается).

**Дефект (не арифметики, а мёртвого knob):** на `block_form="transformer"`
`norm_eps=1e-3` доходит только до финальной нормы; 12 блоков используют torch's
1e-5. Это уже записано в `docs/reviews/2026-11-07-model-internals.md` (V2);
подтверждаю. На loop-форме `norm_eps` доходит до loop-норм — там knob живой.
`mhc.EPS=1e-20` (App. A.1) — другая норма для другого блока, не ошибка, но
два разных «RMSNorm ε» в одном дереве.

---

## 4. Losses

| # | formula | code | source | verdict |
|---|---|---|---|---|
| L1 | byte CE `−log softmax(z)_t`, mean over positions | `model.py:_byte_ce` (`:1499`), `loopblock.py:_iter_ce` | — | **RIGHT** |
| L2 | fixed CE: forward `log1p(s·e^{m−z_t})`, target вне `m`/`s`; backward `q_t ≡ −Σ_{j≠t}q_j` | `losses.py:120-165` | `2610.31642` §2, Eq. 6-7 | **RIGHT** |
| L3 | z-loss `coef·mean(LSE(z)²)` | `model.py:1234` | PaLM §5 / Gemma-2 §9 | **RIGHT** |
| L4 | masked CE mean `Σ ce·w / Σ w` | `loopblock.py:171-181` | doc_masking | **RIGHT** denominator; **NaN** на all-zero маске |
| L5 | `bpb = loss/ln2` | `loop.py:64-66` | — | **RIGHT** для per-byte CE |

**Числа.**

* L2: forward против f64 `cross_entropy` — `max|Δ|` 4.57e-7 / 7.63e-6 / 1.53e-5
  при масштабах логits 1 / 30 / 100; градиент против plain CE — `max|Δg|`
  1.86e-8 / 7.45e-9 / 0.0. «Что меняет fixed»: **только backward у cliff**.
  При margin=40 (target выше остальных): plain loss = **0.0**, plain `g_t` =
  **0.0**, `Σg` = **+1.083e-15**; fixed loss = **1.083e-15**, fixed `g_t` =
  **−1.083e-15**, `Σg` = 4.5e-22. При margin=80: plain `Σg` = 4.60e-33,
  fixed `Σg` = 8.27e-40. Это ровно тот инвариант, который чинит статья.
  Гейт: `|Σ_c g_c| < 1e-12` при margin=80; на plain он 4.6e-33 (не 0, но
  target-компонента ровно 0 — ловится сравнением `g_t` с `−Σ_{j≠t}`).
* L3: `logits=0, V=256` → `LSE=ln256=5.5452`, `LSE²=30.749`; при `coef=1e-4`
  добавка **3.0749e-3**. Если бы стоял `coef·mean(LSE)` без квадрата, было бы
  **5.5452e-4** (в 5.5× меньше) — гейт на пропущенный квадрат.
* L4: маска из 6 единиц: `masked_ce_mean = 0.11659666003821866`, явный
  `ce[mask].mean() = 0.11659666003821861`; `Σw=6`. All-zero маска → **nan**
  (0/0), не `loud`. Инвариант «non-empty» держится ридером, но функция сама
  его не проверяет.
* L5: `bpb(ln 256)=8.0`. Оговорка: headline `bpb` — это **depth-average**
  `(ΣCE_i + CE_fin)/(T+1)/ln2`, а `bpb_fin` — финальный readout; это уже
  разобрано в `docs/reviews/2026-11-06-bpb-accounting.md`. Формула `loss/ln2`
  верна; вопрос, какой `loss` подставлен, — отдельный.

---

## 5. Расписания и EMA

### 5.1 WSD

`wsd_lr` (`lr.py:30-63`):

```
warmup:  lr = peak·(step+1)/warmup            (step < warmup)
hold:    lr = peak
decay:   start = max(total − decay, warmup)
         frac = min(1, (step − start + 1)/decay)
         lr = peak·(1 − frac·(1 − final_frac))
```

Проверено (`total=100, warm=10, decay=20`): `s0=0.1`, `s9=1.0`, `s10=1.0`,
`last=0.1`; при `final_frac=0` last=**0.0**; монотонно после warmup. **RIGHT**
на нормальном домене.

**Граница сломана.** Если `warmup + decay > total`, то `start = warmup`, но
делитель остаётся `decay`, и до `final_frac` расписание не доезжает:

```
wsd_lr(last, total=10, peak=1, warmup=3, decay=20, final_frac=0.1) = 0.685  (не 0.1)
wsd_lr(last, total=10, peak=1, warmup=10, decay=5, final_frac=0.1) = 1.000  (decay не начался)
```

`decay = round(decay_frac·steps)` и `warmup = warmup_steps(warmup_tokens, ...)`
не проверяются на `warm + decay <= steps`. Живые пресеты в норме, но контракт
«decay to the floor» нарушается на коротком прогоне с длинным warmup.
**Fix:** `den = max(1, min(decay, total − start))` или `loud` в `train()` при
`warm + decay > steps`. Гейт: последний шаг обязан быть `peak·final_frac` для
всех `(total, warm, decay)` с `warm + decay <= steps`, и `loud` иначе.

### 5.2 EMA

`_ema_update` (`loop.py:850-871`): `weight = flag·(1−d)`,
`shadow ← lerp(shadow, param, weight)`, т.е. `d·shadow + (1−d)·param`.
`flag=0` на NaN-шаге → shadow не двигается; `d→1` запрещён конфигом
(`config.py:5195`); `d=0` — выключено. Резюме: shadow восстанавливается из
`model_ema` или стартует с загруженных весов. Тест optimizer-suite: значения
0.1, 0.29, 0.561, 0.9049, 1.31441 против closed form — **RIGHT**; 3 шага +
resume == 5 непрерывных бит-в-бит.

**Граница:** resume чекпойнта, записанного с `ema_decay=0` (нет `model_ema`),
с `ema_decay>0` стартует shadow **с текущих весов**, а не с историей траектории.
Это задокументировано; это не ошибка формулы, но EMA тогда — EMA не всего
прогона. Пометить в отчёте прогона.

### 5.3 `min_lr_frac=0` — красный тест

`tests/test_lr_tail.py:141` ждёт `LoudError` на `min_lr_frac=0.0`; `config.py:5092`
разрешает `0.0 <= min_lr_frac <= 1.0`, потому что `final_frac=0` — это
Ripple-65B arm (`wsd_lr` docstring, `lr.py:22-26`). На HEAD тест **красный**:

```
FAILED tests/test_lr_tail.py::test_min_lr_frac_outside_its_range_is_refused[0.0]
```

Либо тест устарел (и тогда его надо переписать: `0` легален + гейт «last=0»),
либо `0` надо запретить (и тогда Ripple-arm невозможен). Сейчас в дереве
противоречие; это единственный красный тест в прогнанном наборе.

---

## 6. Машинерия 1M-контекста (`model/csa.py`)

### 6.1 Что верно

| # | formula | code | verdict |
|---|---|---|---|
| C1 | ring append: `(sum + new)/chunk`, `n·chunk + count == bytes seen` | `_carried_state`, `csa.py:900-970`; assert `:969` | **RIGHT** |
| C2 | `pos = (count + arange(t))//chunk + n`, `key < pos` | `forward_state`, `csa.py:1010-1040` | **RIGHT** |
| C3 | coarse/mid dense level: mean-pool, strictly earlier, softmax | `_level_scores`/`_dense_out`, `csa.py:650-720` | **RIGHT** (алгебра; статью DeepSeek-V4.1 в repo нет — **UNVERIFIED** на fidelity) |
| C4 | hierarchical selection: top-k по уровням, дети выживших | `_hier_fine`, `csa.py:719-795` | **WRONG** (6.2, 6.3) |

**Числа C1/C2.** В моём прогоне (chunk 8, coarse 16, mid 32, t=128, cut=43)
состояние после сегмента: fine `n=5, count=3` → `5·8+3=43`; coarse `2·16+11=43`;
mid `1·32+11=43`. Один и тот же байтовый поток, `forward` vs
`forward_state` (полное окно, иерархический конфиг): `max|Δlogits|` 2.98e-8
(t=100), 0.0 (t=128), 5.96e-8 (t=129), 1.49e-8 (t=200). Два сегмента **flat**
гейтит `tests/test_segment_csa.py` — 13/13, включая жёсткий `segment_len=100`,
`csa_chunk=8`; тест реально проверяет математику (сравнивает с one-shot и
ручным `mean(x[96:104])`), а не только plumbing. Для flat-пути здесь **RIGHT**;
для иерархического 2-сегментный проход — 6.3 (6e-2).

### 6.2 Иерархия падает при `topk < n_parent`

`csa.py:770`:

```python
sel = torch.zeros(b, qn, keep.shape[-1], dtype=torch.bool, device=q.device)
sel.scatter_(2, keep, True)          # keep — индексы В РОДИТЕЛЬСКОМ уровне
allowed = allowed & sel.index_select(2, parent)
```

`keep` — это top-k индексов родительского уровня, их значения лежат в
`0..n_parent-1`. Но `sel` имеет длину `keep.shape[-1] = min(topk, n_parent)`,
а не `n_parent`. Если `topk < n_parent`, `scatter_`/`index_select` выходят за
границу. Это не редкий случай: это **условие работы иерархии** (иначе top-k
ничего не сужает).

Воспроизведено на живом конфиге (`csa_chunk=8, csa_coarse_chunk=256,
csa_coarse_kv_dim=64, csa_mid_chunk=512, csa_topk=32`, `t=16392`, `b=1`):

```
RuntimeError: index 32 is out of bounds for dimension 2 with size 32
```

`t=16392 = 32·512 + 8` — на байт больше `topk · mid_chunk`. `scripts/ab_longctx.py:149`
запускает `csa_mid_chunk=512, csa_hierarchical=true` на 32k окне — там
`ceil(32768/512)=64 > 32`, падение гарантировано. `step_state` (decode)
падает там же, как только mid-ring перерастёт topk. `forward` падает раньше
gather'а, так что это не OOM.

**Cheapest fix (2 строки):**

```python
n_parent = n                      # до overwrite; на первом уровне не нужен
...
sel = torch.zeros(b, qn, n_parent, ...)
parent_ok = allowed.expand(b, qn, n).gather(2, keep)
sel.scatter_(2, keep, parent_ok)
...
ok = valid & (cand < pos.view(1, -1, 1)) & parent_ok.repeat_interleave(ratio, dim=-1)
```

Гейт: `forward` на t=16392 с longctx-конфигом не должен бросать; и
`forward == forward_state` на 2 сегментах (см. 6.3).

### 6.3 Утечка parent-маски: `forward` и `forward_state` — разные функции

`csa.py:785`: `ok = valid & (cand < pos)`. Здесь **нет** проверки, что родитель
кандидата был allowed. В one-shot `keep` содержит и замаскированных родителей
(topk по `finfo.min`), и их дети проходят `cand < pos`, если fine-индекс
меньше `pos`. В ring-пути частичного родителя в кольце нет, поэтому его детей
нет. Следствие: `forward` (one-shot) и `forward_state`/`step_state` (ring)
дают разные выходы на одних и тех же байтах.

Измерено (chunk 8, coarse 16, mid 32, `topk=1000`, t=128, cut=43):

```
one-shot vs 2-сегментный forward_state:  max|Δ| = 5.96e-2  (позиции 40..42)
```

Позиции 40..42 — это хвост fine-chunk 5 (40..47), который в сегменте 1 ещё
частичный. One-shot подтягивает fine-chunk 4 (32..39) через
замаскированного coarse-родителя 2; ring-путь — нет. Разница 6e-2 на выходе
O(6e-2) — это не round-off, это другой набор ключей.

Это же число объясняет assertion в `loopblock.py:1620-1628`: там записано
«`forward_state` и `forward` уже расходятся на 1.7e-2» и на этом основании
`segment_len + csa_hierarchical` **запрещён**. То есть баг известен и
законсервирован отказом, а не исправлен.

**Парадокс теста.** `tests/test_long4m.py::test_gate_3...` утверждает
«hierarchical == flat, когда topk покрывает все уровни» и получает
`max|Δ| = 4.47e-8`. Я применил parent-маску в scratch-копии `_hier_fine`
(без правки repo) и получил:

```
orig (как в коде)  vs flat: max|Δ| = 4.47e-8     # тест зелёный
fixed (parent mask) vs flat: max|Δ| = 0.2961     # 240 позиций
```

То есть тест **проходит только благодаря утечке**. «Hier == flat при
покрывающем topk» — ложное утверждение: иерархия по построению не пускает
дети в собственный coarse/mid-контейнер, а flat пускает. Утечка случайно
восстанавливает часть локального контекста, но зависит от того, лежит ли
частичный родитель в кольце. Это не математика, это состояние.

**Cheapest fix (после 6.2):** сделать набор ключей функцией только позиции:
1) `_mean_pool` для ключей — `ceil_mode=False` (хвостовой частичный чанк не
   может быть прочитан ни одним легальным запросом: его собственные запросы
   его маскируют, более ранние видят будущее, более поздних в окне нет);
2) parent-маска на каждом уровне (код выше);
3) пере-скоупить gate 3: «hier == flat вне первого контейнера каждого уровня»
   или добавить ручную containment-референс-функцию.

**Гейт:** новый тест `t=128/256, chunk=8, coarse=16, mid=32, topk=2`,
cut=43/100: `forward == forward_state` (2 сегмента) до 1e-5; и `forward`
против ручного containment-референса. Сейчас первый кейс **бросает**, второй
даёт 6e-2.

**Фикс проверен в scratch** (monkeypatch на инстансе, repo не менялся):
`sel` по `n_parent` + parent mask на каждом уровне + `ceil_mode=False` дают
на `t=128/256/129, topk=2` `forward == forward_state` до **3.0e-8** и
`forward` vs 2-сегментный `forward_state` до **8.9e-8**; крэша нет. Отдельно
проверено, что `ceil_mode=False` не двигает flat-выход: `ceil` vs `floor` ≤
**4.47e-8** на t=100/129/200 (хвостовой чанк всё равно замаскирован для всех
легальных запросов).

### 6.4 Тест иерархии не тестирует математику

`test_long4m.py` gate 3 использует `topk=64/2` при `n_mid=2` (короткое окно —
`n_mid=0`), т.е. `topk >= n_mid` всегда; крэш 6.2 недостижим. Он сравнивает
hier с flat (который является утечкой), а не с ручной containment-реализацией.
`narrow` arm (`topk=2`) сужает coarse 2 из 4, но mid остаётся 2 из 2 — та же
история.
`test_segment_csa.py` иерархию не трогает вовсе (она запрещена с
`segment_len`). Итог: **иерархическая математика не покрыта ни одним тестом**;
покрыт plumbing (counters, fallback, «не allclose»).

---

## 7. KDA decay (`model/kda.py`)

| # | formula | code | source | verdict |
|---|---|---|---|---|
| K1 | `g = logsigmoid(g_proj(x))`, `β = sigmoid(β_proj(x))` | `kda.py:362-363` | FLA `chunk_kda` (log-space g); K3 Eq. 5 — **другая** параметризация | **RIGHT** для своей формы; **отклонение** от K3 |
| K2 | `S ← Diag(e^g)S + βk(v − kᵀDiag(e^g)S)`, `o = scale·qᵀS` | `kda.py:184-188` | FLA `naive_recurrent_kda` | **RIGHT** |
| K3 | `α = e^g ∈ (0,1]` | `kda.py:362` | — | **RIGHT** сверху; **снизу не ограничено** |
| K4 | q/k L2 norm ε | `_recurrent_step` 1e-6; SDPA fallback `F.normalize` 1e-12 | FLA `l2norm_fwd(eps=1e-6)` | **WRONG** (fallback only) |

**Числа.** `exp(logsigmoid(z)) = sigmoid(z)` до 1.11e-16; диапазон на
`z∈[−200,200]`: `(1.38e-87, 1.0]`; `g.max = −1.38e-87 ≤ 0`. Рекуррентность
против `fla.ops.kda.naive.naive_recurrent_kda` — `max|Δ| = 8.94e-8` при
случайных q/k/v/g/β (q/k нормализованы как в порте). Это **RIGHT**.

**Отклонение от K3.** `docs/papers/gdn-kda.md:147` (Eq. 12):
`g_t = −exp(a) ⊙ softplus(W_f x_t + δ)`, и §2.1.1 Eq. 5 K3:
`g = g_min·sigmoid(e^{A_h} z)`, `g_min=−5`, т.е. `α ≥ e^{−5} = 6.7e-3`
(нижняя граница decay). Rust-крейт это имеет (`G_MIN=-5`, `min_decay` —
`gdn-kda.md` rows 19/23). Python-порт **не имеет**: `logsigmoid(−100) = −100`,
`α = e^{−100} = 3.7e-44`. Это сознательное упрощение (в docstring: FLA-ядро
хочет `A_log`/`dt_bias`, которых нет в параметрах), но оно **не
задокументировано как потеря нижней границы** и не имеет гейта. Последствие:
канал может «забыть всё» за один шаг; K3 вводит floor именно против этого.
**Cheapest fix:** `g = g.clamp_min(-5.0)` под флагом (1 строка) или явно
записать «port's KDA variant: no g_min». Гейт: при `g_proj=−100` `α=3.7e-44`
против `e^{−5}=6.7e-3` — падает на текущем коде.

**K4.** `F.normalize(..., dim=-1)` (`kda.py:672-673`) использует дефолтный
`eps=1e-12`; fused-ядро и `_recurrent_step` — 1e-6. При нормах q/k ~1 разница
1e-6; при коллапсе головы — на порядки. Это только fallback (COUNTED
деградация), но docstring говорит «the same F.normalize», что неточно.
Fix: `F.normalize(..., eps=1e-6)` — одна строка, гейт на q с нормой 1e-4.

---

## 8. Ранжированная таблица

| rank | formula / claim | verdict | blast radius | cheapest fix + gate |
|---:|---|---|---|---|
| 1 | hierarchical CSA `sel` sized `keep.shape[-1]` (`csa.py:770`) | **WRONG** | every `csa_hierarchical=true` run with `t > topk·mid_chunk`: longctx 32k arm, any 1M decode; `forward`, `forward_state`, `step_state` | size `sel` by `n_parent`; gate: t=16 392 longctx config must not raise |
| 2 | hierarchical parent mask missing (`csa.py:785`) | **WRONG** | all hierarchical runs; `forward` != `forward_state` by 6e-2 at cuts inside a coarse chunk; `_forward_segmented` refuses the pair instead of fixing it | parent mask + `ceil_mode=False` keys; gate: `forward == forward_state` over a 2-segment split at t=128/256, topk=2 |
| 3 | «NS-3 = 1.17e-15 from the manifold» (`1b-readiness-plan:280`, `muon-orth-cost:69`, `the-16x:50`, `test_train.py:61`) | **WRONG** | docs and the 3-vs-5 decision; every Muon run's justification | replace with measured 1.7e-1 per-entry; gate: diagonal `p^k(0.5)` + `_ortho_per_entry` on the 10:1 fixture |
| 4 | «3 steps is a tie on rank 64 / not on `[832,3072]`» (`the-16x:49-52`) | **WRONG as stated**; the 0.27 is plausible | the `ns_steps` rung decision; E1-v3 attribution | measure `cos(ColRow NS3,NS5)` on saved momentum per shape; rule: tie iff cos>0.98. Real dense: 0.711; rank-64: 0.985–1.0 |
| 5 | NS prescale `x/(‖x‖+1e-7)` absolute (`optim.py:272,341`) | **WRONG** (scale-invariance) | small/ill-conditioned TSCT factors: `[2048,4]` sv ratio 1.66e7 after NS-5, `[3072,4]` 4.70e7; `flagship` 1e-19 buffers linear | relative floor `x/‖x‖.clamp_min(tiny)`; gate: live `[2048,4]` buffer sv ratio <2 (currently 1.66e7) |
| 6 | WSD `decay` denominator when `warm+decay>total` (`lr.py:60-63`) | **WRONG at boundary** | short runs with long warmup: last LR 0.685 not 0.1; long warmup → no decay | `den=max(1,min(decay,total-start))` or `loud`; gate: last step == peak·final_frac for all legal triples |
| 7 | `min_lr_frac=0` test (`test_lr_tail.py:141`) | **WRONG (test vs code)** | the `final_frac=0` Ripple arm is ungated; repo has a red test | update test to accept 0 + assert last=0 (or forbid 0 and drop the arm) |
| 8 | KDA decay no lower bound (`kda.py:362` vs K3 Eq. 5 `g_min=-5`) | **deviation, ungated** | KDA long-range memory: α can be 3.7e-44 vs paper's 6.7e-3 | `g.clamp_min(-5.0)` behind a flag; gate: `g_proj=-100` → α≥e^-5 |
| 9 | KDA SDPA `F.normalize` eps 1e-12 vs kernel 1e-6 (`kda.py:672`) | **WRONG (fallback)** | CPU / no-FLA fallback near-zero q/k; not the fused arm | `eps=1e-6`; gate: q norm 1e-4 vs hand |
| 10 | `muon_lr_mode` default mismatch (`config.py:2057` uniform vs `optim.py:892/1727` shape_fair) | **WRONG (defaults)** | any direct `Muon`/`MixOptimizer` construction: step 15–55× hot; `Muon(lr=0.02)` 28× published | align defaults; gate: constructor ratio vs `TrainConfig` |
| 11 | ColRow ε placement (`optim.py:486-488` vs paper `sqrt(sum+eps)`) | **RIGHT** live / **deviation** | norms <1e-4 only; not live | document; gate: zero/tiny column |
| 12 | `masked_ce_mean` all-zero mask (`loopblock.py:181`) | **WRONG (guard)** | a padding-only batch returns NaN; the firewall skips the step (counted), but the function itself is silent | `w.sum().clamp_min(1)` + `loud`/count; gate: all-zero mask |
| 13 | `norm_eps` dead on `block_form="transformer"` (`transformer.py:163` vs `config.py:572`) | **WRONG (dead knob)** | dense-form A/Bs that claim to test `norm_eps`; 12 of 13 norms fixed at 1e-5 | wire `cfg.norm_eps` or refuse the knob on the form; gate: `--set norm_eps` moves block output |
| 14 | RoPE / SDPA scale / causal / GQA / softcap / RMSNorm / init / residual / fixed CE / z-loss / EMA / ring carry | **RIGHT** | — | gates are the numbers in §2-§7; each fails on a one-line mutation |

---

## 9. Что осталось непроверенным

* **CSA против DeepSeek-V4.1**: статьи в `docs/papers` нет; проверена
  внутренняя алгебра (mean-pool, strict causal, ring, carry), не fidelity к
  статье. `[U]`.
* **Muon `triton` NS arm**: на CPU — только COUNTED fallback; сам fused-кернел
  не исполнялся. Классическая ветка проверена; `triton`-ветка — `[U]`.
* **EMA на GPU / `opt_graph`**: формулы проверены на CPU и тестами; graph
  replay не исполнялся.
* **`bpb` на ByteFlow/patch**: `loss/ln2` верен для per-byte CE; patch-арм
  требует своего делителя, это отдельный accounting-вопрос
  (`docs/reviews/2026-11-06-bpb-accounting.md`), не проверялся здесь.
* **Полный suite** не гонялся (16 ГБ /home, GPU занят); прогнаны файлы из
  шапки. Единственный красный — `test_lr_tail[0.0]`.
* **`tools/check_doc_refs.py`** красный на **5 pre-existing** путях
  (`the-design.md:23`, `patch-compute.md:152`, `deletion-audit.md:243`,
  `segment-csa.md:277`, `config.py:413`), ни один не из этого отчёта; ссылки
  самого отчёта резолвятся (1250 файлов, 3256 ссылок). Это отдельный
  doc-gate, не математика, но repo не зелёный и по нему.

---

**Итог.** Код Muon, норм, CE, WSD (на нормальном домене), EMA и
KDA-рекуррентности — арифметически верен. Сломаны: иерархический CSA (крэш +
state-зависимая селекция) и документированная ложь про NS-ортогональность, на
которой держится выбор 3-vs-5. Плюс красный тест `min_lr_frac=0` и абсолютный ε
в NS-прескейле.
