# The dispatch guard, and what actually unblocks the fused kernels — 2026-10-02

Lane: снять dispatch-guard — заставить fused-ядра работать в тренировке.
Worktree `wt/dispatch-guard` off `c8df252` (restart with the previous agent's
uncommitted prototype as live state; everything below re-verified from the
tree, not taken from the handover). GPU discipline: owner's fb waves own the
card in ~17-min windows; every cargo under `tools/build_lock.sh`.

## 1. Механика отказа (точно)

`burn-dispatch-0.22.0-pre.4/src/tensor.rs:481-487`
(`DispatchKindConversion::try_into_backend`, macro `impl_dispatch_conversion`):
любая конверсия dispatch-тензора в **bare**-бэкенд требует
`tensor.autodiff == DispatchAutodiffContext::Disabled` и возвращает
`Err("Expected concrete {B} backend with disabled autodiff context, got …")`
иначе. Трейнер живёт на `Autodiff<Cuda, BalancedCheckpointing>` — у всех его
тензоров контекст `Enabled(Balanced)`, поэтому «понизить» тензор до bare-CUDA
невозможно **по дизайну**: bare-результат, обёрнутый обратно в граф, был бы
листом (`8fa5d4c` — дефект-класс «fused forward запустился, градиент потерялся,
loss-кривая здорова»).

## 2. Выбор паттерна: (а) custom-node, не (б), не (в)

Рабочий паттерн уже в дереве — `burn-gdn2::chunk_wy_forward_autodiff_s`
(исправлен и доказан `d8fa449`, `tests/kda_param_grads_cuda.rs`):

1. `autodiff_node::<Inner, S, D>` — поднять тензор до **его собственного**
   autodiff-примитива (`try_into_primitive::<Autodiff<Inner, S>>` — законно:
   мы просим тот же бэкенд, что назван в контексте, а не bare);
2. `bare_from_node` — `Tensor::from_primitive::<Inner>(node.primitive())`
   (те же байты, контекст Disabled);
3. запустить fused-кернел на bare;
4. зарегистрировать вывод кастомным узлом (`Backward::prepare::<S>` +
   `checkpoint` + `finish`) — backward пишет градиенты через
   `grads.register::<B>` на bare-бэкенде `B = Inner`.

**Почему не (б) (апстримный фикс в burn-dispatch):** новый вариант контекста
«разрешить понижение с трекингом» — это ровно то, что делает кастом-узел, но
внутри burn, с тем же риском листа и на чужой территории. Правило репо — не
конфликтовать с вендором без крайней нужды; узел локален, безопасен и уже
доказан. **Почему не (в):** ручной fused-rewrite целиком (ADR-0009) был
медленнее burn — урок тот же: сначала счётчики конкретного узла (§6).

Осторожность, которую нельзя терять: `is_require_grad()` — не тот вопрос
(`requirement == Grad` — строгий лист; входы трейнера всегда
`GradInBackward`). Вопрос — `is_tracked()`.

## 3. KDA fused backward: CUDA-гейт — ЗЕЛЁНЫЙ, прогнан из этого дерева

«fused adjoint is wrong and non-deterministic» (§3.3 AGENTS) устарела: починка
kda-adjoint lane (`439d8c8` — двухчанковый гейт, в HEAD) закрывает её.
Прогон **этим агентом** в этом worktree (debug, vendor-воркспейс, CUDA,
`cargo test -p burn-kda --features cuda,autodiff --test kda_param_grads_cuda
-- --nocapture --test-threads=1`, 435 s, 2026-10-02):

- **4/4 зелёные**; все 11 групп на тренерной форме + 3 conv-группы на
  short-conv форме: `non_zero=true finite=true`, `cpu_rel=0.00e0`
  (бит-в-бит против CPU-autodiff), FD-вердикты выведены (худший
  разрешённый rel 4.9e-4, бар 5e-2; две decay-группы UNRESOLVED по
  шумовой модели прибора и несут cpu-референс на баре 5e-3 — как
  задокументировано в шапке теста).
- Проводка в трейнере уже живая и правок не требует: `loop_block.rs:832` →
  `KdaModule::forward_train_state` → `kda_fused_chunk_reported::<B>` →
  `burn_gdn2::chunk_dispatch` (стратегия — параметр). Kill switch:
  `DM_FUSED_KDA=0`.

## 4. Прототип: rmsnorm как один autodiff-узел — ГОТОВ И ЗЕЛЁНЫЙ

Ядро работает на bare с `34c5631` (1.168e-7), но в тренировке — `norm=0/N`
во всех логах: `rmsnorm_cuda` просит bare-понижение и получает отказ. Сделано
по паттерну §2 (файлы: `burn-rmsnorm/src/ops.rs`, `src/cuda_dispatch.rs`,
`src/fused.rs`, `src/lib.rs`):

- `rmsnorm_node_autodiff_s::<Inner, S>` — 2 родителя (x2d, weight), оба
  чекпоинтятся; backward — тензорный на bare
  (`dx_i = w_i·inv·gy_i − x_i·inv³/d·Σ_j w_j·gy_j·x_j`, `dw_i = Σ_r gy·x·inv`);
- `rmsnorm_node_autodiff` пробует обе стратегии (Balanced → No) — стратегия
  живёт в типе, модуль не знает стратегию вызывающего; отказ — типокомпаре;
- `RMSNorm::forward`: ОДИН ask на вызов → узел → bare-launch → тензорный
  путь; kill switch `DM_RMSNORM_FUSED=0`; счётчик руки `arm_counts() ->
  (asked, skipped, node_ran)`; `norm=ran/asked` eval-строки считает узел без
  правок трейнера (ran = asked − skipped).

**Гейты зелёные (CUDA, этим агентом, 2026-10-02):** `autodiff_node_cuda`
3/3, `fused_kernel_gate` 3/3 — включая ПЕРЕВЁРНУТЫЙ пин
(`the_fused_kernel_takes_the_node_on_an_autodiff_device`: один ask, ноль
скипов, NODE_RAN, backward доходит до входа). Градиенты узла против
тензорного пути на тренерном бэкенде (`Autodiff<Cuda, Balanced>`,
родители-проекции = форма трейнера): **dx 1.3e-7, dw 1.1e-7
scale-relative**, бар 1e-6.

**Две находки в фикстуре (обе — урок о приборе, не о коде):**

1. **Градиент GradInBackward-узла ПОТРЕБЛЯЕТСЯ его собственным шагом**
   (`burn-autodiff src/grads.rs:103-107`: consume = `remove`, у листа —
   `get`). `x.grad()` на промежуточном тензоре после здорового backward —
   `None`; первый прогон фикстуры упал там, пока узел работал. Читать
   градиенты на `require_grad()`-листьях.
2. **Per-element relative error не имеет f32-оценки на малых элементах:**
   abs 4.8e-7 на элементе 1.6e-3 при масштабе тензора 47 = 3e-4 per-element,
   но 1e-8 scale-relative. Бар — scale-relative (`max|a−b|/max|ref|`),
   1e-6 ≈ 4000× под f32 eps: неверный знак/член пересекают его мгновенно.
   И данные фикстуры детерминированы (device RNG не сидирован — §3.3).

## 5. Счётчики запусков (до/после) — априори

Тензорный forward RMSNorm = 6 запусков (powf, mean, add, sqrt, div, mul);
узел = 1. Backward: тензорный ~8-10, узел ~10 тензорных + (fused adjoint —
follow-up). Тренер делает ~3.1 norm-вызова/шаг (1560 asks / 500 шагов, §3.3
AGENTS) → экономия ~15 запусков/шаг из тысяч. Ожидание по ADR-0009: выигрыш
<1% по шагу; решает измерение (§6), не априори. KDA fused backward —
отдельная статья: bwd attention был ~221 ms тензорного replay, fused adjoint
— 2 запуска на чанк вместо ~150.

## 6. Измерение на тренере (150 шагов, small b8 s512 d4, fp32, --no-engram, aux off, seed 1)

Две руки одним бинарем + атрибуционная третья. Машина тихая (нет cargo/train
параллельно), systemd-run MemoryMax=40G, `--timers`, показания шагов 50/100
(шаг-0 — автотюн-артефакт, §3.1), окно eval 81 920 B у всех рук.

**Находка, объясняющая всю историю `fused kda=f/0`:** тип трейнера
(`Backend = Autodiff<Cuda, BalancedCheckpointing>`, train/src/lib.rs:35)
противоречил его же девайсу (`Device::cuda(0).autodiff()` = контекст
`Enabled(Disabled)` = NoCheckpointing). Burn'овские операции type-erased —
тип не влиял ни на что, КРОМЕ швов, называющих тип в конверсии:
`chunk_dispatch`' финальный cast `try_into_primitive::<B>` отказывал на
**каждом** тренировочном KDA-вызове — fused forward запускался и
ВЫБРАСЫВАЛСЯ в ops-путь каждый шаг. Проба до фикса (`dg2_ab_f1`):
`fused kda=412/0 asked=492 declined=1064 ops=492 node_bwd=0` — 412 fused
форвардов в мусор, 100% градиентов из ops-графа. Арифметика сходится
точно: 492 asks = 412 train (узел строился, cast падал) + 80 eval
(untracked → decline); 1064 = 412×1 + 80×3 declines; eval — 20 батчей ×
depth 4.

**Фикс (train/src/lib.rs): тип = NoCheckpointing** — тип перестал лгать о
том, что реально исполнялось во всех прогонах проекта. Balanced остаётся
опцией владельца (тогда `.gradient_checkpointing()` на девайсе + тип
одновременно — это смена числового пути, отдельный A/B).

**После фикса (`dg2_ab_f3`), все руки тихой машиной, бинарь один:**

| рука | ms/step (50/100) | launches/step (50→100) | fwd | bwd | eval-строка |
|---|---|---|---|---|---|
| control: всё тензорное (`DG_*_FUSED=0`) | 394 / 449 | 15 644 | ~102 | ~240 | `fused kda=0/0 ops=492 norm=0/615` |
| fused KDA + tensor rmsnorm (`DG_RMSNORM_FUSED=0`) | 434 / 473 | 8 432 | ~35 | ~319 | `fused kda=412/412 node_bwd=412 ops=80 norm=0/615` |
| fused KDA + rmsnorm узел | 427 / 438 | 8 072 | ~31 | ~318 | `fused kda=412/412 node_bwd=412 ops=80 norm=515/615` |

(контроль dg2_ab_t4, руки dg2_ab_f4 / dg2_ab_f3; bpb 6.497 / 6.503 / 6.514 —
внутри сид-шума на 150 шагах; численная эквивалентность adjoint'а
огорожена гейтами §3, ≤4.7e-7.)

**Вердикты:**
- **KDA fused backward ВКЛЮЧЁН и РАБОТАЕТ**: `fused kda=412/412`,
  `node_bwd=412` — первые ненулевые fused-бэкварды в истории проекта на
  живом тренировочном прогоне. Accounting точный, ноль silent fallbacks.
- **launches/step: 15 644 → 8 072 (−48%)** — валюта CUDA-graph программы:
  выгода replay ≈ launches − 1, т.е. графовый потолок этой формы удвоился.
- **fwd −70% (102→31 ms)**; **bwd +33% (240→318 ms)**: fused adjoint делает
  реальную арифметику, а тензорный bwd на batch 8 — launch-bound (GPU простаивает
  87%, §3.1), поэтому «~150 запусков на чанк» в wall-clock дешевле, чем 2
  тяжёлых ядра. На b8 netto ~+2.6% (внутри разброса 40-55 ms шага одной
  руки) — **выигрыш <5% по правилу брифа: задокументировано**. На больших
  батчах (compute-bound зона) расклад должен перевернуться — не мерил.
- **rmsnorm узел**: 515/615 запусков в тренировке (100 = eval-скипы,
  посчитанные: eval-тензоры несут `Enabled`-контекст при невалидном графе,
  bare-рука отказывает в demotion, узел declines по untracked → тензорный
  путь — COUNTED, ответ верный). Вклад: −360 launches/step (3.43 вызова/шаг
  × ~105 тензорных опов), fwd −4 ms, в ms/step неразличим. Выигрыш <5% —
  задокументировано; fused adjoint-кернел для rmsnorm backward — follow-up.
- Старый заголовок `kda_param_grads_cuda.rs` говорил «трейнер живёт на
  Balanced» — исправлен (§1.7); оба пина стратегии в файле сохранены.

## 7. Что осталось (follow-ups, не эта lane)

- fused adjoint-кернел для rmsnorm backward (~10 запусков → 1-2);
- attnres / mhc — тот же узловой паттерн;
- eval-скипы rmsnorm (100/615): bare-рука могла бы брать untracked
  `Enabled`-тензоры через `try_strip` (как gdn2) — мелочь, eval-only;
- замер fused KDA на compute-bound форме (b32+) — где adjoint должен
  выигрывать в wall-clock, а не только в launches;
- quant_probe (example) называет Balanced на дефолтном девайсе — тот же
  latent mismatch, пример-only;
- решение владельца: Balanced-режим (память) как отдельный A/B, если
  нужен.

## 8. Два дефекта прибора, найденных при закрытии lane (2026-10-02, вечер)

1. **`tools/lib_gate.sh:95` глотает красный главной ячейки.**
   `[ "$rc" -eq 0 ] || rc=$rc2` ПЕРЕЗАПИСЫВАЕТ ненулевой rc ячейки
   `cargo test --workspace` значением rc2 (burn-gdn2 binary-tests). Красная
   ячейка 2 + зелёная ячейка 3 = зелёный гейт. Измерено: мой локальный
   прогон напечатал `PASS lib_gate` при 6 FAILED прямо в логе (kda_oracle
   3 designed-red + burn-spectral 3 known-red), а solo-прогон
   `cargo test -p burn-kda --test kda_oracle` в том же дереве честно
   выходит 101. Фикс — удалить строку 95 (строка 96 уже покрывает rc2).
   **Не правил** (общий инструмент; переплетено с designed-red решением
   владельца) — сообщено с file:line. Следствие: локальный «зелёный» гейт
   этой lane был ложно-зелёным; честное состояние дерева = CI
   fused-library, красный по designed-red тестам.

2. **fused-library CI красный на MAIN** (3/3 последних прогона main
   failure, те же тесты): burn-kda kda_oracle `read_scale_matches_fla_reference`
   / `chunked_wy_applies_no_read_scale` /
   `kimi_linear_softplus_decay_matches_fla_reference` — designed-red
   (владельческое решение по scale=1.0, docs/reviews/2026-09-30-kda-formula-audit.md
   §3.2); фикс `#[ignore = "<причина>"]` лежит в `wt/kdaci` (`23f4b5f`) и
   НЕ смержен в main; burn-spectral `polar_retracts` + два inference —
   known-red из шапки gpu-gate.sh. Моя ветка не добавила ни одного нового
   красного: workflow `ci` (корневой) — зелёный; fused-library падает
   идентично main.
