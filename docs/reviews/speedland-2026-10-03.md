# speedland lane findings — 2026-10-03

Lane: посадить готовые скорости (wt/speedland от main `6637ee9`).
GPU не трогали (night2e b8 на карте): все гейты — CPU/lib tests +
review уже замеренных lane-A/B.

## 1. Head-wise Q/K NS batching — ПОСАЖЕНО

Cherry-pick `c442060` (wt/maxopt2) → `perf(train): batch the head-wise Q/K
Newton-Schulz`. Один stacked NS над `[n_heads, dh, cols]` вместо 12
отдельных matmul-цепочек; `orthogonalize_batched`,
`crates/dormouse-train/src/optim.rs`.

Gate: `cargo test -p dormouse-train --lib optim` — 7/7 зелёный, включая
`headwise_batched_ns_matches_per_head`.

Ожидаемый эффект: −15 ms/step по atlas-оценке wt/maxopt2
(opt-стадия, launch-bound). Точного quiet-card замера ms/step в этой
lane не гоняли — GPU занят night2e.

## 2. Retraction cadence — дефолт НЕ меняем, вердикт

A/B maxopt2 (2k steps, b8 s512 d2 Fp16, window 81920 B):
re1 6.415/6.350 (s1/s2) | re5 6.426/6.372 | re10_s1 6.433 —
все внутри спреда контроля 0.065..0.083. BPB-вош. ms-фавор re5
(−40 ms/step est) не подтверждён: чтения с contention-пометкой,
quiet-card repeat не сделан, re10_s2 не добежал (lane отменена).

Решение по брифу: каденс банкуют только с A/B-подтверждением 3k+;
3k не гоняли → `retract_every` остаётся 1 в дефолте
(`TrainCfg::default`, проброшено тестом
`cfg_retract_every_default_is_one`). Quiet-card ms-repeat re5 и/или
3k-прогон — follow-up для очереди A/B, не для silent-дефолта.

## 3. dispatch-guard lane — состояние wt/dispatch-guard

- `feece22` — фикс Backend-типа (cuda Backend называл
  `BalancedCheckpointing`, а девайс строил `NoCheckpointing`): ПОСАЖЕН
  отдельным коммитом. Evidence lane: `~/logs/dg2_ab_f{3,4}.log`,
  `fused kda=412/412 node_bwd=412 ops=80 declined=240` (первый живой
  backward фьюжн-пути в истории проекта), adjoint equivalence ≤4.7e-7,
  net wall +2.6% = шум. `cargo check -p dormouse-train --features cuda`
  зелёный (CPU compile, без GPU-прогона).
- `ec438ec` (rmsnorm fused kernel как один autodiff node) — НЕ сел:
  vendor-изменения burn-rmsnorm/burn-gdn2, требует собственного
  CUDA-гейта rmsnorm node arm; follow-up (в докладе dispatch-guard
  fused rmsnorm backward kernel ещё ожидается).
- Два дефекта прибора lane (lib_gate.sh:95 глотает красный;
  fused-library CI красный на main из-за designed-red) — зафиксированы
  в докладе, не чинили (lib_gate — общий инструмент, решение владельца).

## 4. benches/history.tsv

Только аппенд; ни sort, ни checkout --theirs.
