# ThunderKittens-дизайн на cubecl: гейт, что переносится, план

Дата: 2026-10-04. Источник: TK 2.0 (blog + README), разбор нашего `vendor/cubecl-fix`.
Решение владельца: **не FFI/C++, а их дизайн на cubecl** — развиваем свою экосистему.

## Гейт (нашёл при разборе бэкенда)

Наш CUDA-бэкенд — pliron→LLVM→NVPTX. `restrict_to_llvm_backend`
(`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:475-480`) **явно выключает**
ровно те примитивы, из которых состоит TK:

```rust
// No TMA, no clusters, no async copy, and no `mbarrier` behind them.
props.features.tma = Default::default();
props.features.cube_cluster = false;
props.features.copy_async = false;
props.features.types.opaque.remove(&OpaqueType::Barrier);
```

TMA, clusters, cp.async и mbarrier — это **двигатель** TK (асинхронный
producer/consumer pipeline). В нашем бэкенде он снят.

## Но — вот что меняет всё

**IR уже говорит на языке TK.** `cubecl-ir/src/dialect/barrier.rs` содержит
полный набор: `MemCopyAsyncOp`, `CopyAsyncOp`, `ArriveOp`,
`ArriveAndExpectTxOp`, `CommitCopyAsyncOp`, `ExpectTxOp`, `WaitOp`,
`WaitParityOp`, `ArriveAndWaitOp`. Плюс `dialect/asm.rs` — inline-PTX escape.

То есть **фронтенд/IR умеет**, а **LLVM-бэкенд не умеет** — понижения этих ops
в NVPTX нет ни в `cubecl-cuda`, ни в `cubecl-cpp` (grep пуст). Фичи
*анонсируются* для C++-пути (`runtime.rs:237-263`: `Barrier`, `copy_async`,
`tma`, `cube_cluster = true`), но `cpp`-фича пустая и off (§2).

**Вывод: «их дизайн на cubecl» — это не DSL с нуля, это проект понижения в
бэкенде.** Словарь есть, семантика есть, не хватает `cp.async.*`/`mbarrier.*`
в NVPTX-лоуэринге. Мы владеем этим кодом (`vendor/cubecl-fix`).

## Что переносится на sm_120

| техника TK | железо sm_120 | наш бэкенд | вердикт |
|---|---|---|---|
| tcgen05 / tensor memory | ✗ (sm_100) | ✗ | мертво |
| WGMMA | ✗ (sm_90) | ✗ | мертво |
| TMA | ✓ | ✗ выключен | **работа в бэкенде** |
| cp.async + mbarrier | ✓ (sm_80+) | ✗ выключен | **работа в бэкенде** |
| clusters / DSMEM | ✓ | ✗ выключен | **работа в бэкенде** |
| mma.sync (f16) | ✓ | ✓ анонсирован | есть |
| tile-примитивы + типизация layout | ✓ | ✓ (типы/comptime/shared) | **строим сами** |
| persistent grid | ✓ | ✓ (grid/block dims) | **строим сами** |
| warp specialization | ✓ | ✓ на уровне source | строим, но без async — слабо |
| `elect.sync` | ✓ | ? | спайк |
| megakernel | ✓ | ✓ (persistent + обычный barrier) | долгосрочно |
| benchmark-дисциплина | ✓ | ✓ | **забрать сейчас, бесплатно** |

## План (по стоимости × эффект)

```
0. СЕЙЧАС, бесплатно
   - TK benchmark-конвенция (bitwise-identical inputs, input groups против L2,
     500 warmup, 100 iters, 2 events, cooldown) — до 10% разброса методики.
     Применяем ко ВСЕМ нашим kernel-A/B.
   - CUDA graphs (§3.1, 5/5 зелёных) — уже бьёт в launch overhead БЕЗ переписывания ядер.

1. БЭКЕНД (наш экосистемный бет, ограниченный)
   - Понижение CopyAsync/Barrier ops → NVPTX (cp.async.*, mbarrier.*).
     IR уже есть; пишем лоуэринг в cubecl-cuda(LLVM). Проверка: спайк-ядро
     с async-пайплайном на sm_120.
   - Это разблокирует весь TK-словарь.

2. ЦЕЛЬ — KDA fwd/bwd
   - Самая большая рука модели (80% атрибуции, отозвана — но крупнейшая).
   - fused adjoint СЕЙЧАС НЕВЕРЕН (§3.3, d8fa449): 4.5e-2…2.6e-1 расхождение.
     Сначала корректность, потом персистентность/warp-spec.
   - A/B против tensor-op fallback (25.8 s/step batch-8, число непроверено).

3. НЕ трогать: GEMM-throughput. Мы уже 91–97% cuBLAS, GEMM = 2–5% шага.
```

## Честный скоуп и риск

- Работа 1 — компиляторная (pliron/LLVM-диалект), не «написать ядро». Недели,
  не дни. Но она **bounded**: ops и семантика уже в IR, нужен лоуэринг.
- Работа 2 без работы 1 возможна (persistent grid + tile-слой + mma.sync), но
  без async-пайплайна это ~20–40% дизайна TK, не двигатель.
- Реальный bottleneck по замерам — **launch overhead (GPU 87% простоя)**, не
  throughput ядра. Работа 0 (graphs) бьёт в него сегодня; работа 1+2 — в
  скорость KDA, где сейчас ещё и корректность.
- Kill-switch: если async-лоуэринг не даёт измеримого выигрыша на KDA против
  tensor-op пути — ядро не окупается, откат.
