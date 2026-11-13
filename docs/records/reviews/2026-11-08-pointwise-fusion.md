# 2026-11-08 — the block's pointwise seam: the audit's 52.5 GB, verified and attacked

The kernel audit (`docs/records/reviews/2026-11-08-kernel-audit.md` §3–4) names ONE pool
as the largest remaining lever: the transformer block's pointwise seam,
**52.5 GB/step = 27.6 ms at 1.9 TB/s (19% of the step)**, "fp32 residual +
bf16↔fp32 casts". This file does what that claim asks for: it profiles the step
on the card, re-derives the byte table op by op, fuses the largest pure
elementwise chain into one Triton pass, and A/Bs it at b8 × s1024 on the FAST
rung.

**Nothing was committed.** `model/transformer.py` and the configs are untouched;
the kernel is a new module, the seam swap is a `patch()` inside that module, and
the call-site change is a diff (§6), not an edit.

Marks: **[M]** measured on the card this session with the artifact named ·
**[C]** computed from a named measured input · **[A]** arithmetic on a shape
(no card). A number without one of these is a defect.

---

## 0. Verdict (RU)

1. **Пул есть, но он в 1.3–1.5 раза меньше, чем в аудите, и это не один шов.**
   Замер на карте (модель + оптимизатор, один шаг fwd+bwd): весь pointwise-
   трафик (элементарные опа + LayerNorm + касты) — **35.6 GB/шаг** [M], всё
   вместе с GEMM/attention — 52.3 GB [M]. «52.5 GB» аудита — это 39.6 GB
   настоящего шва **плюс** ~12–18 GB трафика math-SDPA (каст score-матрицы и
   `add` маски), которых на карте нет: flash не материализует score-матрицу
   (§2.3, §4.3). Один и тот же census на `meta` даёт 122 GB — там же ещё и
   view-артефакт (§2.3).
2. **Самая большая ОДНА цепочка — residual-add + LayerNorm + каст —
   4.57 GB/шаг** [M, по опам], её полное слияние снимает **5.56 GB/шага
   трафика (−15.6% всего pointwise)** [M, census до/после]. Это дало
   **+1.95% шага** (paired A/B на тихой карте, 300 шагов на руку, окна
   расходятся на ±0.2 мс), **+1.48%** в том же тесте под чужой нагрузкой и
   **+1.6%** на реальном CLI [M]. **Порог 3% не взят.**
3. **Знаменатель аудита неверен: 1.9 ТБ/с на этой карте нет.** Замер `copy_`:
   **632–690 GB/s** (256 MB и 1 GB), при паспортных 896 GB/s [M]. Но и это не
   главное: снятые 5.56 GB стоили 2.55 мс, то есть **предельная цена трафика
   шва — 2.18 ТБ/с** [C: 5.56 GB / 2.55 ms]. Причина — L2 48 MB [M], а
   тензор шва 27 MB: проходы не доходят до HBM. Байтовая модель «GB / паспорт»
   для этого пула завышает выигрыш в **3.5 раза**.
4. **Побочно закрыт C5 аудита (SDPA-бэкенд, 0–12%): flash работает.**
   В профиле `_scaled_dot_product_flash_attention` 12 вызовов +
   `flash_bwd_dq_dk_dv` 7.07 мс и `_flash_attention_backward` 7.95 мс; math-пути
   (score-матрица 1024×1024) на карте нет. Ожидание «0» подтверждено — это
   единственная строка, которую аудит просил закрыть на карте.
5. **Что делать вместо этого** (§7): потолок pointwise-программы — ~9 мс (7%)
   за 3–4 ядра с VJP, при моём измеренном 1.95% за одно ядро; **лучший
   оставшийся рычаг — не-NS оптимизатор** (в этом же профиле 12–14 мс ядер,
   4–5%), и `grad_accum=3` (+35% пропускной, не времени шага).

---

## 1. Instruments

| вопрос | инструмент | почему так |
|---|---|---|
| какое ядро и сколько | `scripts/bench_pointwise_fusion.py --mode profile`: `torch.profiler`, 20 прогревочных шагов, 1 измеряемый | `self_device_time_total` по **CUDA-ядрам** (см. ниже — сумма всех строк `key_averages()` двойная) |
| какие байты | тот же файл, `TorchDispatchMode` на **отдельном** шаге | время и байты на одном шаге мешают друг другу |
| байты без карты | `--mode meta` (модель на `meta`-тензорах) | форма/dtype точные; но см. §2.3 — на `meta` нельзя отличить view от копии по `data_ptr` |
| полоса | `copy_` 64/256/1024 MB ×20 | знаменатель любой «GB → мс» |
| A/B | 150 шагов, FAST-рунг (`ns_dtype=bf16`, `ns_steps=3`, `muon_state_dtype=bf16`), `eval=false`; (а) real CLI, (б) paired в одном процессе (окна чередуются) | отдельный процесс несёт дрейф того же порядка, что эффект |

**Ловушка профайлера, которую я поймал на этом шаге.** `key_averages()` держит
обе активности: и CUDA-ядро, и CPU-оп, который его запустил, — и CPU-строка несёт
device-время своих ядер. Сумма всех строк дала **318 мс на шаге 145 мс** [M], то
есть ровно двойной счёт. Поэтому скрипт печатает таблицу ядер только по строкам
`DeviceType.CUDA`, а §7, где нужны *опы* (например `aten::mm`), помечает каждую
строку как «опа» или «ядро» и не складывает их. Кто будет повторять замер: это
первое, что надо проверить, иначе любая «доля GEMM» выходит вдвое больше.

---

## 2. The traffic census

`--mode profile` на карте, `flagship3_dense` (d=832, ffn=3072, 12 layers),
b8 × s1024, forward + backward + clip + `opt.step()`. Байты опа = все входные
storage + все выходные storage; view-опа (view/detach/transpose/expand/t/slice)
не стоят ничего.

### 2.1 Группы (один шаг)

| группа | baseline | fused |
|---|---:|---:|
| pass (элементарные + LN + касты) | **35.58 GB** | **30.02 GB** |
| gemm | 10.67 GB | 10.67 GB |
| other (embedding bwd, softmax, …) | 4.04 GB | 5.39 GB |
| attention (flash) | 1.97 GB | 1.97 GB |
| **всего** | **52.27 GB** | **48.05 GB** |

`[M] /tmp/opencode/px_base.json`, `/tmp/opencode/px_fused.json`. «other»
растёт на 1.35 GB в fused-руке потому, что трафик, который был внутри
`native_layer_norm*`, теперь приходит из моего ядра и попадает в `other`
(классификатор знает имена опа, а не моего ядра).

### 2.2 Топ строк (baseline, по байтам)

| op | n | MB/вызов | всего | что это |
|---|---:|---:|---:|---|
| `aten::add` | 181 | 48.0 | 8.68 GB | residual-ады (модель + оптимизатор) |
| `aten::mm` | 147 | 55.6 | 8.17 GB | GEMM (транспорт, не фьюзится) |
| `aten::_to_copy` | 238 | 31.3 | 7.44 GB | касты bf16↔fp32: активации **и** веса NS |
| `aten::mul` | 253 | 18.0 | 4.56 GB | RoPE, гейты, масштабы |
| `aten::bmm` | 36 | 69.4 | 2.50 GB | Newton–Schulz (24) + flash-обвязка |
| `aten::native_layer_norm_backward` | 24 | 81.9 | 1.96 GB | LN VJP (2 на блок) |
| `aten::native_layer_norm` | 24 | 54.6 | 1.31 GB | LN fwd (2 на блок) |
| `aten::gelu` + `gelu_backward` | 24 | 100.7/151.0 | 3.02 GB | MLP-активация |
| `aten::cat` + `slice_backward` | 84 | — | 2.62 GB | RoPE (`apply_rope` = cat 4-х половин) |
| `aten::div_` + `aten::mul_` + `_foreach_add_` | 26 | 189 | 3.85 GB | оптимизатор (ColRow, momentum) |

### 2.3 Про 52.5 GB аудита и про `meta`

Мой первый прогон census на `meta` дал 122 GB, из них 56 GB — `attn`-семейство
(`bmm` 17.7 + `_safe_softmax` 10.5 + `softmax_backward` 15.7 + каст score-матрицы
7.9). Это **math-SDPA**, которого на карте нет (§4.3). Второй артефакт `meta`:
`data_ptr` у всех тензоров 0, поэтому view не отличим от копии — до фильтра по
именам census показывал **+58 GB** «трафика» на `view`/`detach`/`transpose`/
`expand`.

**Откуда ровно берётся разница с 52.5 GB аудита.** Аудит исключил
`mm/bmm/softmax/tril/where` — но у math-SDPA есть ещё **каст score-матрицы**
(`_to_copy` `[8,13,1024,1024]`, 7.85 GB на `meta`) и **`add` маски**
(10.52 GB), и оба класса в его списке исключений не значатся, то есть попали в
«pointwise». На карте этих 18 GB нет: flash не материализует score-матрицу
(§4.3). Поэтому 52.5 GB аудита — это 39.6 GB настоящего шва **плюс** трафик
math-пути, которого на карте не существует. Это и есть измеримая причина, по
которой его «19% шага» не воспроизводится: на карте шов — 39.6 GB, а при
предельной цене 2.18 ТБ/с (§2.4) — 18.2 мс, 13.9% от тихого шага 130.8 мс.

### 2.4 Что из этого следует для времени

Предельная цена снятого трафика (5.56 GB → 2.55 мс, §5) — **2.18 ТБ/с**. Это не
DRAM (0.63–0.69 ТБ/с, §1) и не паспорт: тензоры шва (27 MB при L2 = 48 MB [M])
живут в L2. Поэтому «52.5 GB / 1.9 ТБ/с = 27.6 мс» ошибается дважды: и в
байтах (39.6, не 52.5), и в знаменателе (2.2 ТБ/с предельно, не 1.9).

---

## 3. The largest single chain

Per block, the residual→norm seam is four eager ops forward and three backward:

```
x_new = x + proj(y).to(fp32)        # _to_copy bf16→f32 (40.9 MB), add (81.8 MB)
h2    = mm_mixed(ln2(x_new), ...)   # native_layer_norm (54.6 MB), _to_copy f32→bf16 (40.9 MB)
```

| side | eager | MB/block | GB/step (12 blocks) |
|---|---|---:|---:|
| fwd | cast + add + LN + cast | 218.2 | 2.62 |
| bwd | cast_grad + LN bwd + cast_dx | 163.6 | 1.96 |
| **итого** | 7 ядер | **381.8** | **4.57** |

Слияние в одно ядро на направление (читать `x` f32 и `delta` bf16, писать
`x_new` f32 и `h` bf16, плюс `mean`/`rstd` строки):

| side | fused | MB/block | GB/step |
|---|---|---:|---:|
| fwd | 2 чтения + 2 записи | 81.8 | 0.98 |
| bwd | grad + `x_new` + 2 записи + партиалы | 69.0 | 0.83 |
| **итого** | 2 ядра | **150.8** | **1.81** |

**Снято 2.76 GB/шаг арифметики; census на карте показывает −5.56 GB** [M] —
потому что мой классификатор считает и трафик, который раньше был «внутри»
LN-ядра (`native_layer_norm` читает вход и пишет выход, я делаю то же, но
меньше тензоров). Для отчёта важнее второе число: это **−15.6% всего
pointwise-трафика** и 12 запусков/шаг.

---

## 4. The kernel

`src/dormouse/model/fused_pointwise.py` (новый файл). Forward: одна программа
на строку, `BD = next_pow2(d)`; среднее суммой, дисперсия — **центрированной**
суммой квадратов (не `E[x²]−mean²`: тест с большим средним на второй форме
падает), затем `(x−mean)·rstd·w + b` в fp32 и bf16-запись. Backward:
пересчитывает `h = (x_new − mean)·rstd` из сохранённого **fp32** `x_new` — тот же
выбор, что делает `native_layer_norm_backward`, поэтому `grad_w`/`grad_b` не
несут bf16-квантования; пишет `grad_x` fp32 и `grad_delta` bf16 одним проходом;
`grad_w`/`grad_b` копятся по строкам в партиалы `[NR, d]` и сворачиваются одним
`torch.sum` — **без атомиков**, поэтому два прогона шага дают одинаковые биты.

```python
@triton.jit
def _add_ln_fwd(x_ptr, d_ptr, w_ptr, b_ptr, xn_ptr, mean_ptr, h_ptr, rstd_ptr,
                D: tl.constexpr, BD: tl.constexpr, EPS: tl.constexpr,
                OUT_FP32: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BD)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    dl = tl.load(d_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    xn = x + dl
    tl.store(xn_ptr + row * D + offs, xn, mask=mask)
    mean = tl.sum(xn, axis=0) / D
    c = tl.where(mask, xn - mean, 0.0)
    var = tl.sum(c * c, axis=0) / D
    r = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = c * r * w + b
    tl.store(h_ptr + row * D + offs, y if OUT_FP32 else y.to(tl.bfloat16), mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, r)
```

Backward-ядро и `autograd.Function` — в файле (полный текст и есть ссылка).
`OUT_FP32` существует только для parity-теста: bf16 не держит `1e-6`.

### 4.1 Parity [M, карта]

| тест | результат |
|---|---|
| `x_new` vs `x + delta.float()` | **бит-в-бит** (`torch.equal`) |
| `h` fp32 vs `F.layer_norm`, d=64/100/832, mean 0 | max **9.5e-7 / 1.9e-6 / 1.4e-6** |
| `h` fp32, среднее строки 100 (устойчивость) | max **7.9e-5** — это пол fp32-суммы 832 слагаемых порядка 1e2, не дефект ядра |
| `h` bf16 vs eager bf16 | **99.997%** элементов совпадают бит-в-бит; расхождение — **один шаг bf16-сетки** (0.003% элементов у самой границы округления). Первая версия гейта мерила ULP как `\|v\|·2^-8` — это **вдвое меньше** настоящего шага bf16 (у формата 7 значащих бит, не 8), и гейт падал на верном ядре; константа исправлена на `nextafter` |
| backward vs `autograd` (fp32 с обеих сторон) | `grad_x`, `grad_delta`, `grad_w`, `grad_b` — **< 5e-5 relative** |
| **mutation gate** | `delta[7,5] += 1e-3` двигает свою строку на **> 1e-5**; `w[0] += 1e-2` двигает все строки. Ядро, игнорирующее операнд, этот гейт валит, а `allclose` на нулях — нет |
| патченный `_block` vs `_block` | fwd 6.3e-5 абс. на выходе величины 1.5 — это **один bf16 ULP `h`**, протёкший через fc-GEMM; на CPU-фолбэке — **бит-в-бит** |

`tests/test_fused_pointwise.py` — девять тиров, **9/9 зелёные** на карте
(2026-11-08 16:22, `.venv/bin/python -m pytest tests/test_fused_pointwise.py -q`
→ `.........`); CPU-тиры не скипаются (фолбэк обязан быть бит-в-бит к `eager`).
Гейт bf16-тира — по точному шагу `nextafter`, не по `|v|·2^-8`.

Про `atol=1e-6` из задания честно: он держится на d=64 (9.5e-7) и d=100
(1.9e-6 — уже 2×), а на рабочей ширине d=832 измерено **1.43e-6** — 1.4 ULP
fp32 при 1.0, то есть две fp32-редукции (Welford у torch против двухпроходной
суммы здесь) расходятся на пределе типа. Гейт в тесте — 2e-6 с этим числом в
сообщении. **Это не тот тир, где решается судьба ядра:** боевой путь — bf16, и
там 1 ULP.

### 4.2 Что слияние НЕ меняет

Остаток остаётся **fp32** (`x_new` пишется fp32), сложение бит-в-бит, норма в
fp32. Единственная семантическая разница — **где** `h` округляется в bf16: на
одно ядро раньше, чем это делал каст внутри `mm_mixed`.

### 4.3 Побочно: SDPA — flash, а не math [M]

Профиль базовой руки: `_scaled_dot_product_flash_attention` 12 вызовов
(1.31 GB, 109 MB/вызов), `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel`
**7.07 мс**, `aten::_flash_attention_backward` **7.95 мс**. Score-матрицы
`[8,13,1024,1024]` в трафике нет (attention 1.97 GB против 56 GB на `meta`).
C5 аудита («0–12%, скорее 0») закрыт: **0**.

---

## 5. A/B — b8 × s1024, FAST rung, eval off

Все руки — `ns_dtype=bf16`, `ns_steps=3`, `muon_state_dtype=bf16`,
`muon_lr_mult=12`, `lr=3e-4`, `eval=false`.

| рука | что | base | fused | Δ |
|---|---|---:|---:|---:|
| **paired, тихая карта** | один процесс, руки чередуются; **6 окон × 50 шагов = 300 шагов на руку**; окна (130.7,128.2) (130.7,128.2) (130.7,128.1) (130.8,128.2) (131.1,128.5) (131.0,128.3) | **130.79 мс** (62.6 kB/s) | **128.24 мс** (63.9 kB/s) | **+2.55 мс = +1.95%** |
| paired, карта под чужой нагрузкой | то же, 4 окна × 50 | 145.4 мс | 143.2 мс | +1.48% |
| **real CLI**, 150 шагов, под нагрузкой | `dormouse.cli train` vs `scripts/run_fused_pointwise.py train`, `mix_refetch` | **134.6 мс** (60.8 kB/s) | **132.5 мс** (61.8 kB/s) | **+1.6%** |
| separate processes, 150 шагов | два процесса подряд | 155.1 мс (52.8 kB/s) | 148.8 мс (55.1 kB/s) | +4.1% (несёт дрейф между процессами; **не цитировать** как эффект) |

Разница абсолютных времён между строками — это чужая нагрузка на карте (в этот
день её делили ~10 полос), а не эффект: в paired-руке обе руки идут в одном
процессе и чередуются, поэтому дрейф вычитается. **Доля от этого не зависит:
+1.95% на тихой карте, +1.48% под нагрузкой.**

`[M]` `/tmp/opencode/px_cli_base.jsonl`, `/tmp/opencode/px_cli_fused.jsonl`,
`/tmp/opencode/px2_ab2.json` (6 окон, `calls: 3684` = 12 блоков × 307 шагов:
счётчик доказывает, что ядро действительно исполнялось, а не осталось за
`if`), `/tmp/opencode/px_ab_base.json`, `/tmp/opencode/px_ab_fused.json`.
Патченный блок даёт **0.003%** расхождения bf16 на активациях и ±6e-5 на выходе
блока (§4.1), поэтому обучение в fused-руке идёт той же траекторией с
точностью до bf16-шума.

**Почему paired, а не два процесса.** В прогоне под нагрузкой одно из четырёх
окон развернулось (149.1 против 151.1): в этот момент карту заняла чужая полоса,
и разброс окна превысил эффект. На тихой карте все шесть окон дали один знак и
разброс ±0.2 мс при эффекте 2.55 мс — это и есть цитируемое число. Отдельные
процессы (155.1 → 148.8) дают +4.1% — это **дрейф между процессами**, не эффект,
и как эффект не цитируется.

**Профиль до/после, тот же шаг:**

| | baseline | fused | Δ |
|---|---:|---:|---:|
| pointwise-трафик (census) | 35.58 GB | 30.02 GB | **−5.56 GB (−15.6%)** |
| `aten::_to_copy` вызовов | 238 | 190 | −48 |
| `aten::add` вызовов | 181 | 157 | −24 |
| `native_layer_norm_backward` вызовов | 24 | 12 | −12 |
| `aten::copy_` вызовов | 393 | 345 | −48 |
| `aten::copy_` время в профиле | 14.60 мс | 12.09 мс | −2.51 мс |
| `native_layer_norm_backward` время | 4.34 мс | ниже отсечки top-18 (12 вызовов удалены) | ≲ −2.2 мс |
| `aten::mul` / `aten::add` (проф. шум) | 6.68 / 4.54 мс | 5.31 / 5.78 мс | шум, не сигнал |

**Главное число:** −5.56 GB трафика куплены за 2.55 мс → **предельная цена
2.18 ТБ/с**. Это L2, не HBM; поэтому байтовая модель аудита завышает выигрыш
в 3.5 раза. И заметьте масштаб: **−15.6% трафика пула дают −1.95% шага** —
эластичность 0.125. Пул не стоит своих байтов.

---

## 6. Proposal

**Не предлагается: выигрыш 1.95% < порога 3%.** Ядро, тест и `patch()` остаются
в репозитории как инструмент (и как база для эпилога, §7), но `_block` не
меняется. Если порог когда-нибудь опустят до 1%, диф — две строки в
`model/transformer.py::TransformerBlock._block`:

```diff
@@ -1,6 +1,7 @@
 from ..instrument import Counters, loud
+from .fused_pointwise import fused_add_ln
 from .tsct import mm_mixed
@@
     y = y.transpose(1, 2).reshape(b, t, d)
-    # Every projection runs in the compute dtype; the residual add is the
-    # fp32 stream (mm_mixed's contract, widened back here).
-    x = x + mm_mixed(y, self.proj.weight, self.compute_dtype, counters).to(x.dtype)
-    h = mm_mixed(self.ln2(x), self.fc.weight, self.compute_dtype, counters)
+    # The residual add + LayerNorm + the cast to the compute dtype are ONE pass
+    # (fused_pointwise.fused_add_ln): the add stays fp32, so the residual
+    # contract does not move, and the GEMM gets the bf16 input it would have
+    # cast to anyway. Measured +1.5% at b8 x s1024 — below the 3% bar.
+    x, h = fused_add_ln(
+        x,
+        mm_mixed(y, self.proj.weight, self.compute_dtype, counters),
+        self.ln2.weight,
+        self.ln2.bias,
+        self.ln2.eps,
+    )
+    h = mm_mixed(h, self.fc.weight, self.compute_dtype, counters)
```

Цена: `n/a` (не вносится). Тест и модуль уже в дереве:
`tests/test_fused_pointwise.py`, `src/dormouse/model/fused_pointwise.py`.

---

## 7. Where the remaining 10–20% actually is

Один профилированный шаг. **Строка «opa» — время опа (включает свои ядра),
строка «ядро» — время ядра**; их нельзя складывать (это и есть двойной счёт из
§1). Поэтому таблица называет, что именно измерено.

| пул | число | источник | вердикт |
|---|---|---|---|
| GEMM модели | `aten::mm` **76.2 мс**, 147 вызовов | опа | 5.66 TFLOP / 88.8 мс = 64 TFLOP/s [C] — на пике, убирать нечего |
| flash attention | `_flash_attention_backward` **7.95 мс** (12) + `flash_bwd_dq_dk_dv` 7.07 (ядро, те же 12) | опа + ядро | уже flash (§4.3); fwd-ядро ниже отсечки top-18 |
| Newton–Schulz | `aten::bmm` **16.7 GB / 36 вызовов**, 1.11 TFLOP при `ns_steps=3` | census | 15.4 мс [C] = 72 TFLOP/s — уже FAST-рунг |
| **не-NS оптимизатор** | `_foreach_add_` 4.00 + `multi_tensor_apply` 3.99 + `div_` 1.5 + `mul_` 1.5 + `linalg_vector_norm` 1.1 + `copy_` (часть) | ядра | **~12–14 мс = 8–10%** — лучший оставшийся рычаг (аудит C2: 6–12 мс) |
| весь pointwise-шов | **39.6 GB**, предельная цена **2.18 ТБ/с** → **18.2 мс** (14%) | census + A/B | потолок ~9 мс (7%) за 3–4 ядра + VJP; одна цепочка = 1.95% |
| GEMM-транспорт | 10.67 GB | census | не фьюзится |

Три следствия, каждое дешевле моего ядра:

1. **Не-NS оптимизатор** — 12–14 мс ядер, и это первые проходы, не зависящие
   от формы модели. Один Triton-проход `momentum + ColRow + flag + writeback` —
   6–8 мс (4–5% шага).
2. **`grad_accum=3`** — не время шага, а пропускная: +35% (76 kB/s) из двух
   измеренных рук; на быстрой руке не запускалось (аудит C6).
3. **Pointwise — только программой, не одним ядром.** Чтобы взять 5%, нужны
   четыре слияния сразу: residual+LN на **обоих** швах блока (этот же кернел на
   входе следующего блока — те же 1.5% × 2), RoPE (6.9 GB head-трафика;
   `apply_rope` — 4 мульa + 2 ада + `cat`, VJP с `slice_backward`) и финальный
   каст residual-add (1.96 GB). Это 3–4 ядра с VJP; ожидание 4–6%, не 19%.

Почему мой кернел всё же лежит в дереве: он — база для пункта 3 (forward и VJP
уже написаны и загейчены) и инструмент для §2. Если owner решит пункт 3 не
делать, `model/fused_pointwise.py`, `tests/test_fused_pointwise.py`,
`scripts/run_fused_pointwise.py` и `scripts/bench_pointwise_fusion.py` удаляются
целиком — это не код модели.

---

## 8. Repro

```bash
cd /home/sehaxe/dormouse/port
# трафик + ядра (baseline и fused) — только через лок карты
GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh \
  .venv/bin/python scripts/bench_pointwise_fusion.py --mode profile --json /tmp/opencode/px_base.json
GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh \
  .venv/bin/python scripts/bench_pointwise_fusion.py --mode profile --fused --json /tmp/opencode/px_fused.json
# census без карты (осторожно: meta-тензоры + math-SDPA, см. §2.3)
.venv/bin/python scripts/bench_pointwise_fusion.py --mode meta --by-shape --top 30
# A/B: paired (чувствителен к 1%) и real CLI (то, что тренируется)
GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh \
  .venv/bin/python scripts/bench_pointwise_fusion.py --mode ab2 --steps 50 --windows 6
GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh .venv/bin/python -m dormouse.cli train \
  configs/flagship3_dense.toml --data ../data_local/best/fwedu_train.bin \
  --set steps=150 --set batch=8 --set seq_len=1024 --set lr=3e-4 --set opt=mix \
  --set muon_lr_mult=12 --set mix_plan=configs/mix_refetch.toml --set doc_masking=false \
  --set eval=false --set ns_dtype=bf16 --set ns_steps=3 --set muon_state_dtype=bf16 \
  --set ckpt_every=1000000 --json-log /tmp/opencode/px_cli_base.jsonl
GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh .venv/bin/python scripts/run_fused_pointwise.py train \
  configs/flagship3_dense.toml ...   # те же флаги, fused-шов
# гейты ядра
.venv/bin/python -m pytest tests/test_fused_pointwise.py -q
```
