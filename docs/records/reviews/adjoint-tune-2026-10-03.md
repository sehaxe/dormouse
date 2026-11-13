# adjoint-tune lane findings — 2026-10-03

Lane: сделать правильный fused KDA adjoint БЫСТРЫМ + аллокатор (wt/adjoint-tune от `7c442ce`).
GPU: night2e fp32 b8 владеет картой (шаг ~15k/80k на момент lane), quiet-card
200-шаговая рука ms/step bwd НЕ гонялась — заметочный долг для следующей quiet-карты;
все CUDA-гейты гоняли мелкие батчи на той же карте (безопасно, night2e жив).

## 1. Профиль adjoint — откуда launches

Atlas по коду (`DM_LAUNCH_ATLAS=1`-счётчики per-stage имели намение в `atlas.rs`;
вот per-call разбор `fused_chunk_backward`, `chunk_adjoint_cube.rs`):

- BK2 (`gdn2_chunk_inter_adjoint_kernel`) — 1 launch; пишет `d_v_new`, траекторию `d_s`.
- BK1 (`gdn2_chunk_intra_adjoint_kernel`) — 1 launch.
- **Тензорный клей между ними — ~14 launches/call**: `d_s_shift` cat (копировала
  всю чанк-траекторию [b,h,nt,k,v]), `swap_dims(1,2)*1.0` (вторая такая же копия),
  `repeat`, `div` для decay, batched matmul `d_hat`, три `d_k_bptt/d_decay/d_e_bptt`
  цепочки (6+ элементарных ядер), `sum_dim`, плюс `zeros` на 10+ буферах.
- `d_e_last` вычислялось в двух проходах (включая full-trajectory `states`-mul и
  `sum_dim`) и было **мертвым**: его коэффициент уходил только в ядро BK1 как
  конвейерный член d_g, но по смыслу являлся частью BK2-цепочки.
- Клей — это и есть "медленность" adjoint'a: в debug-свопне каждый launch = полный
  прогон; в release — ~14×24µs ≈ 0.34ms на call × 2 calls/шаг (depth-2 ветка).

## 2. Тюнинг — что сделано (10e657c)

- **BPTT-клей вклеен в BK1**: ядро читает прямой экспорт `v_new` (уже был
  аргументом), `d_s` (выход BK2) и `glast`, считает `d_hat[r][k]=Σ_v v_new·d_s_next`,
  `d_k_bptt=dh·decay`, `d_e_bptt=-dh·k·decay/E` инлайн. `d_e_last` сводится в
  эпилоге d_g (staged partials + `d_s_next⊙s_before` член) — тот же член, тот же
  порядок редукции по c/v, но без trajectory-cat и второго прохода.
- Удалены: `d_s_shift` cat, `swap_dims*1.0`, `decay` repeat/div, matmul `d_hat`,
  6 элементарных цепочек клея, буферы `d_k_bptt/d_e_bptt/d_e_last`,
  `d_v_new_fresh`-копия (алиасинг-страх не подтверждается: d_v_new жив в живом
  дереве аллокатора до конца call).
- `zeros`→`empty` на 8 выходных буферах BK2/BK1 (все полностью перезаписываются ядрами).

Числа про probe `tests/adjoint_alloc_gate.rs` (b=8,h=12,t=512,k=64,v=64,c=16,
one fwd + one fused bwd, warm, live GPU shared with night2e):

| | backward wall (debug) | allocs | in_use | reserved pool |
|---|---|---|---|---|
| до | 13.996 ms | 31 | 328.5 MB | 865.0 MB |
| после | 1.680 ms | 28 | 328.5 MB | 603.9 MB |

`in_use` не изменился: живые буферы байт-в-байт те же, выигрыш — ликвидированные
временные копии траектории и мёртвая математика d_e_last (это и есть -261 MB
reserved, ~-30% пул). Debug wall не переносится на release; заказной замер
ms/step bwd на 200-шаговой руке — долг, quiet-card.

## 3. Гейты (тёмная карта, debug build, night2e рядом)

- `fused_adjoint_matches_the_ops_path_on_cuda` — worst rel **6.305e-7** на d_g
  (4-chunk), one-chunk worst **4.789e-7** — не растёт vs предыдущего ≤4.7e-7.
- `fused_chunk_verify` 3/3, `kda_param_grads_cuda` 4/4 (включая fused-arm grad flow).
- CPU-гейты vendor не запускали в этой lane (night2e no-CPU-contention policy:
  сборки гнали одну, монотонно).

## 4. Аллокатор fwd — что сознательно НЕ делали

Pre-аллокация 17 fresh-тензоров итерации в scratch-структуру НЕ даёт
VRAM-relief: `IntraOut` (aqk, w, u, kgd, glast, qgt, wvt, m_inv, v_new,
states, out, gexp) удерживается до backward'а, поэтому переиспользовать их
внутри одного fwd нельзя; транзиенты (kgt/bkt/akk) и так уходят в cubecl pool
при free и переиспользуются тем же размером. Реальный OOM-класс порезался
ликвидацией trajectory-cat'а и мёртвых промежуточных d_s_shift/swap*1.0
(~-261 MB reserved на probe-форме, см. таблицу) — это и есть пункт 3 брифа.
`v_new_out/states_out` остаются +262144-паддингом по уникальности размера для
alloc-trace; удаление паддинга — отдельный patch, не брал.

## 5. Чем оплачено / follow-ups

- Debug probe 8.9× — не число production bwd; заказной quiet-card прогон
  (200 steps, `--timers` step%50) нужен для ms/step bwd before/after.
- night2e шаг ~15k/80k; агент НЕ трогал; pgrep/nvidia-smi перед каждым GPU-тестом.
- `DM_LAUNCH_ATLAS=1` на короткой руке с пер-стадийным срезом KDA-vs-total
  остался на когда вечером карта свободна.
